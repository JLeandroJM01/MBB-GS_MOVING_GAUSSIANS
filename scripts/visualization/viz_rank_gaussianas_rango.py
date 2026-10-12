import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint


@torch.no_grad()
def eval_series(a0, hi, B):
    return torch.cat([a0, hi], dim=-1) @ B.T


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--inicio", type=int, required=True)
    ap.add_argument("--fin", type=int, required=True, help="Frame final INCLUSIVO")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    salida = Path(args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(args.checkpoint, device=args.device)

    N = int(info["N"])
    T = int(info["n_frames"])
    H = int(info["H"])
    W = int(info["W"])

    inicio = max(0, int(args.inicio))
    fin = min(int(args.fin), T - 1)
    sample_every = max(1, int(args.sample_every))

    if inicio > fin:
        raise RuntimeError(f"Rango invalido: inicio={inicio}, fin={fin}, T={T}")

    cols_t = torch.arange(inicio, fin + 1, sample_every, device=modelo.mu_a0.device)
    T_eff = int(cols_t.numel())

    B_mu = matrices_base[modelo.grados["mu"]][cols_t]
    B_op = matrices_base[modelo.grados["opacity"]][cols_t]
    B_co = matrices_base[modelo.grados["color"]][cols_t]
    B_sc = matrices_base[modelo.grados["scale"]][cols_t]

    log_min = float(np.log(0.5))
    log_max = float(np.log(max(H, W)))

    rows = []

    print("=== rank_gaussianas_rango ===", flush=True)
    print(f"checkpoint : {Path(args.checkpoint).resolve()}", flush=True)
    print(f"salida     : {salida}", flush=True)
    print(f"N          : {N}", flush=True)
    print(f"T total    : {T}", flush=True)
    print(f"rango      : {inicio}..{fin} inclusivo", flush=True)
    print(f"T eval     : {T_eff}", flush=True)
    print(f"chunk      : {args.chunk}", flush=True)

    for s in range(0, N, int(args.chunk)):
        e = min(s + int(args.chunk), N)

        mu = eval_series(modelo.mu_a0[s:e], modelo.mu_high[s:e], B_mu)
        op = torch.sigmoid(eval_series(modelo.opacity_a0[s:e], modelo.opacity_high[s:e], B_op).squeeze(1))
        col = torch.sigmoid(eval_series(modelo.color_a0[s:e], modelo.color_high[s:e], B_co))
        sc = torch.exp(eval_series(modelo.scale_a0[s:e], modelo.scale_high[s:e], B_sc).clamp(min=log_min, max=log_max))

        if T_eff >= 2:
            dmu = mu[:, :, 1:] - mu[:, :, :-1]
            step = torch.sqrt((dmu * dmu).sum(dim=1).clamp_min(0.0))
            path_length = step.sum(dim=1)
            speed_mean = step.mean(dim=1)
            speed_max = step.max(dim=1).values

            dcol = col[:, :, 1:] - col[:, :, :-1]
            color_step = torch.sqrt((dcol * dcol).sum(dim=1).clamp_min(0.0))
            color_path = color_step.sum(dim=1)
        else:
            M = e - s
            path_length = torch.zeros(M, device=modelo.mu_a0.device)
            speed_mean = torch.zeros(M, device=modelo.mu_a0.device)
            speed_max = torch.zeros(M, device=modelo.mu_a0.device)
            color_path = torch.zeros(M, device=modelo.mu_a0.device)

        move_total = torch.sqrt(((mu[:, :, -1] - mu[:, :, 0]) ** 2).sum(dim=1).clamp_min(0.0))

        op_mean = op.mean(dim=1)
        op_max = op.max(dim=1).values
        op_std = op.std(dim=1)
        active_frac = (op > 0.1).float().mean(dim=1)

        color_std = col.std(dim=-1).mean(dim=1)

        scale_mean = sc.mean(dim=-1)
        area_mean = (scale_mean[:, 0] * scale_mean[:, 1]).clamp_min(1e-12)
        scale_std = sc.std(dim=-1).mean(dim=1)

        score_mov_vis = path_length * op_mean * (active_frac + 1e-6)

        # Proxy de importancia visual:
        # favorece gaussianas visibles, activas y con huella espacial relevante.
        score_importancia_visual = (0.7 * op_mean + 0.3 * op_max) * (0.5 + active_frac) * torch.sqrt(area_mean)

        for k in range(e - s):
            rows.append({
                "id": int(s + k),
                "score_mov_vis": float(score_mov_vis[k].detach().cpu()),
                "score_importancia_visual": float(score_importancia_visual[k].detach().cpu()),
                "path_length_px": float(path_length[k].detach().cpu()),
                "move_total_px": float(move_total[k].detach().cpu()),
                "speed_mean_px": float(speed_mean[k].detach().cpu()),
                "speed_max_px": float(speed_max[k].detach().cpu()),
                "op_mean": float(op_mean[k].detach().cpu()),
                "op_max": float(op_max[k].detach().cpu()),
                "op_std": float(op_std[k].detach().cpu()),
                "active_frac": float(active_frac[k].detach().cpu()),
                "color_path": float(color_path[k].detach().cpu()),
                "color_std": float(color_std[k].detach().cpu()),
                "area_mean": float(area_mean[k].detach().cpu()),
                "scale_std": float(scale_std[k].detach().cpu()),
            })

        if e == N or (s // int(args.chunk)) % 10 == 0:
            print(f"  procesadas {e}/{N}", flush=True)

    rows_mov = sorted(rows, key=lambda r: r["score_mov_vis"], reverse=True)
    rows_imp = sorted(rows, key=lambda r: r["score_importancia_visual"], reverse=True)

    write_csv(salida / f"stats_rango_{inicio}_{fin}.csv", rows)
    write_csv(salida / "top30_mov_vis.csv", rows_mov[:30])
    write_csv(salida / "top30_importancia_visual.csv", rows_imp[:30])
    write_csv(salida / "top200_mov_vis.csv", rows_mov[:200])
    write_csv(salida / "top200_importancia_visual.csv", rows_imp[:200])

    with open(salida / "resumen.txt", "w", encoding="utf-8") as f:
        f.write("Ranking de gaussianas por rango temporal\n")
        f.write("========================================\n")
        f.write(f"checkpoint={Path(args.checkpoint).resolve()}\n")
        f.write(f"rango={inicio}..{fin} inclusivo\n")
        f.write(f"T_eval={T_eff}\n")
        f.write("score_mov_vis = path_length_px * op_mean * active_frac\n")
        f.write("score_importancia_visual = (0.7*op_mean + 0.3*op_max) * (0.5 + active_frac) * sqrt(area_mean)\n")

    print("Archivos generados:", flush=True)
    print(f"  {salida / 'top30_mov_vis.csv'}", flush=True)
    print(f"  {salida / 'top30_importancia_visual.csv'}", flush=True)
    print(f"  {salida / 'top200_mov_vis.csv'}", flush=True)
    print(f"  {salida / 'top200_importancia_visual.csv'}", flush=True)
    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
