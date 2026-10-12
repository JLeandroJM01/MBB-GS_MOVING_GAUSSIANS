"""
Genera estadisticas por gaussiana temporal desde un checkpoint.

Objetivo:
    No depender de mirar gaussianas al azar. Este script te dice cuales:
      - se mueven mas
      - cambian mas de color
      - cambian mas de opacidad
      - son visibles pero casi estaticas

Uso Windows, desde la raiz del repo:
    python scripts/visualization/viz_gaussian_stats.py --checkpoint outputs/EXP/checkpoints/checkpoint_final.pt

Opcional para ahorrar RAM/tiempo:
    python scripts/visualization/viz_gaussian_stats.py --checkpoint outputs/EXP/checkpoints/checkpoint_final.pt --chunk 2048 --sample_every 2

Salidas:
    <exp>/viz_stats/gaussian_stats.csv
    <exp>/viz_stats/top_movimiento.csv
    <exp>/viz_stats/top_color.csv
    <exp>/viz_stats/top_opacidad.csv
    <exp>/viz_stats/top_estaticas_visibles.csv
    <exp>/viz_stats/resumen.txt
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint


@torch.no_grad()
def _eval_series(a0, hi, B):
    return torch.cat([a0, hi], dim=-1) @ B.T


def _std_mean_dim(x):
    # x: [M,D,T] o [M,T]
    if x.dim() == 2:
        return x.std(dim=1)
    return x.std(dim=-1).mean(dim=1)


@torch.no_grad()
def calcular_stats(checkpoint, salida=None, chunk=4096, sample_every=1, device="cpu", top=200):
    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(checkpoint, device=device)
    N = info["N"]
    T = info["n_frames"]

    if salida is None:
        salida = Path(checkpoint).resolve().parent.parent / "viz_stats"
    else:
        salida = Path(salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    # Submuestreo temporal opcional: usa 1 de cada k frames para estadisticas.
    if sample_every <= 1:
        cols_t = slice(None)
        T_eff = T
    else:
        cols_t = torch.arange(0, T, int(sample_every), device=modelo.mu_a0.device)
        T_eff = int(cols_t.numel())

    B_mu = matrices_base[modelo.grados["mu"]]
    B_op = matrices_base[modelo.grados["opacity"]]
    B_co = matrices_base[modelo.grados["color"]]
    B_sc = matrices_base[modelo.grados["scale"]]
    B_th = matrices_base[modelo.grados["theta"]]

    if sample_every > 1:
        B_mu = B_mu[cols_t]
        B_op = B_op[cols_t]
        B_co = B_co[cols_t]
        B_sc = B_sc[cols_t]
        B_th = B_th[cols_t]

    log_min = float(np.log(0.5))
    log_max = float(np.log(max(info["H"], info["W"])))

    filas = []
    print("=== viz_gaussian_stats ===", flush=True)
    print(f"checkpoint    : {Path(checkpoint).resolve()}", flush=True)
    print(f"salida        : {salida}", flush=True)
    print(f"N             : {N}", flush=True)
    print(f"T             : {T}  T_eval={T_eff}  sample_every={sample_every}", flush=True)
    print(f"chunk         : {chunk}", flush=True)

    for s in range(0, N, int(chunk)):
        e = min(s + int(chunk), N)

        mu = _eval_series(modelo.mu_a0[s:e], modelo.mu_high[s:e], B_mu)          # [M,2,T]
        op = torch.sigmoid(_eval_series(modelo.opacity_a0[s:e], modelo.opacity_high[s:e], B_op).squeeze(1))  # [M,T]
        col = torch.sigmoid(_eval_series(modelo.color_a0[s:e], modelo.color_high[s:e], B_co))  # [M,3,T]
        sc = torch.exp(_eval_series(modelo.scale_a0[s:e], modelo.scale_high[s:e], B_sc).clamp(min=log_min, max=log_max))  # [M,2,T]
        th = _eval_series(modelo.theta_a0[s:e], modelo.theta_high[s:e], B_th).squeeze(1)  # [M,T]

        if T_eff >= 2:
            dmu = mu[:, :, 1:] - mu[:, :, :-1]
            step = torch.sqrt((dmu * dmu).sum(dim=1).clamp_min(0.0))  # [M,T-1]
            path_length = step.sum(dim=1)
            speed_mean = step.mean(dim=1)
            speed_max = step.max(dim=1).values

            dcol = col[:, :, 1:] - col[:, :, :-1]
            color_step = torch.sqrt((dcol * dcol).sum(dim=1).clamp_min(0.0))
            color_path = color_step.sum(dim=1)
        else:
            M = e - s
            path_length = torch.zeros(M)
            speed_mean = torch.zeros(M)
            speed_max = torch.zeros(M)
            color_path = torch.zeros(M)

        move_total = torch.sqrt(((mu[:, :, -1] - mu[:, :, 0]) ** 2).sum(dim=1).clamp_min(0.0))
        op_mean = op.mean(dim=1)
        op_max = op.max(dim=1).values
        op_std = op.std(dim=1)
        active_frac = (op > 0.1).float().mean(dim=1)

        color_std = _std_mean_dim(col)
        scale_std = _std_mean_dim(sc)
        theta_std = th.std(dim=1)

        dynamic_score = path_length * op_mean
        color_score = color_path * op_mean
        opacity_score = op_std * op_mean
        static_visible_score = op_mean / (path_length + 1e-6)

        for local_i in range(e - s):
            i = s + local_i
            filas.append({
                "id": int(i),
                "op_mean": float(op_mean[local_i]),
                "op_max": float(op_max[local_i]),
                "op_std": float(op_std[local_i]),
                "active_frac": float(active_frac[local_i]),
                "move_total_px": float(move_total[local_i]),
                "path_length_px": float(path_length[local_i]),
                "speed_mean_px": float(speed_mean[local_i]),
                "speed_max_px": float(speed_max[local_i]),
                "color_std": float(color_std[local_i]),
                "color_path": float(color_path[local_i]),
                "scale_std": float(scale_std[local_i]),
                "theta_std": float(theta_std[local_i]),
                "dynamic_score": float(dynamic_score[local_i]),
                "color_score": float(color_score[local_i]),
                "opacity_score": float(opacity_score[local_i]),
                "static_visible_score": float(static_visible_score[local_i]),
            })

        if s == 0 or e == N or (e // int(chunk)) % 10 == 0:
            print(f"  procesadas {e}/{N}", flush=True)

        del mu, op, col, sc, th

    campos = list(filas[0].keys()) if filas else []
    ruta_all = salida / "gaussian_stats.csv"
    with open(ruta_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)

    def guardar_top(nombre, key, reverse=True):
        ruta = salida / nombre
        top_filas = sorted(filas, key=lambda r: r[key], reverse=reverse)[:int(top)]
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(top_filas)
        return ruta, top_filas

    r_mov, top_mov = guardar_top("top_movimiento.csv", "dynamic_score", True)
    r_col, top_col = guardar_top("top_color.csv", "color_score", True)
    r_opa, top_opa = guardar_top("top_opacidad.csv", "opacity_score", True)
    r_sta, top_sta = guardar_top("top_estaticas_visibles.csv", "static_visible_score", True)

    resumen = salida / "resumen.txt"
    with open(resumen, "w", encoding="utf-8") as f:
        f.write("Resumen gaussianas temporales\n")
        f.write("=============================\n")
        f.write(f"N={N}\nT={T}\nT_eval={T_eff}\nsample_every={sample_every}\n\n")
        for titulo, lista in [
            ("Top movimiento", top_mov[:20]),
            ("Top color", top_col[:20]),
            ("Top opacidad", top_opa[:20]),
            ("Top estaticas visibles", top_sta[:20]),
        ]:
            f.write(titulo + "\n")
            f.write("-" * len(titulo) + "\n")
            for r in lista:
                f.write(
                    f"id={r['id']} op_mean={r['op_mean']:.4f} "
                    f"path={r['path_length_px']:.2f} move_total={r['move_total_px']:.2f} "
                    f"color_path={r['color_path']:.4f} op_std={r['op_std']:.4f}\n"
                )
            f.write("\n")

    print("=== listo ===", flush=True)
    print(f"stats completo : {ruta_all}", flush=True)
    print(f"top movimiento : {r_mov}", flush=True)
    print(f"top color      : {r_col}", flush=True)
    print(f"top opacidad   : {r_opa}", flush=True)
    print(f"top estaticas  : {r_sta}", flush=True)
    print(f"resumen        : {resumen}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--sample_every", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top", type=int, default=200)
    args = ap.parse_args()

    calcular_stats(
        checkpoint=args.checkpoint,
        salida=args.salida,
        chunk=args.chunk,
        sample_every=args.sample_every,
        device=args.device,
        top=args.top,
    )


if __name__ == "__main__":
    main()
