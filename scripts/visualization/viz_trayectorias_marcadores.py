"""
Trayectorias y marcadores de gaussianas.

Uso normal:
    python scripts/visualization/viz_trayectorias_marcadores.py --checkpoint outputs/EXP/checkpoints/checkpoint_final.pt --modo top --n 20 --fondo_dir outputs/EXP/frames_renderizados --gif

Usar gaussianas con mas movimiento segun stats:
    python scripts/visualization/viz_trayectorias_marcadores.py --checkpoint ... --ids_csv outputs/EXP/viz_stats/top_movimiento.csv --n 20 --gif --fondo_dir outputs/EXP/frames_renderizados

Usar ids manuales:
    python scripts/visualization/viz_trayectorias_marcadores.py --checkpoint ... --ids 123 456 789 --gif
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint, cargar_fondo_desde_dir_o_gt


@torch.no_grad()
def _evaluar(a0, hi, B):
    return torch.cat([a0, hi], dim=-1) @ B.T


@torch.no_grad()
def _trayectorias_opacidad_color(modelo, matrices_base):
    g_mu = modelo.grados["mu"]
    g_op = modelo.grados["opacity"]
    g_co = modelo.grados["color"]

    mu_t = _evaluar(modelo.mu_a0, modelo.mu_high, matrices_base[g_mu])  # [N,2,T]
    op_t = torch.sigmoid(_evaluar(modelo.opacity_a0, modelo.opacity_high, matrices_base[g_op]).squeeze(1))  # [N,T]
    color_t = torch.sigmoid(_evaluar(modelo.color_a0, modelo.color_high, matrices_base[g_co]))  # [N,3,T]
    return mu_t, op_t, color_t


def _leer_ids_csv(path, n):
    ids = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ids.append(int(row["id"]))
            if len(ids) >= int(n):
                break
    return torch.tensor(ids, dtype=torch.long)


def _seleccionar(mu_t, op_t, modo, n, bbox, semilla, ids=None, ids_csv=None):
    N = mu_t.shape[0]

    if ids_csv:
        idx = _leer_ids_csv(ids_csv, n)
        return idx[(idx >= 0) & (idx < N)]

    if ids:
        idx = torch.tensor([int(x) for x in ids], dtype=torch.long)
        return idx[(idx >= 0) & (idx < N)][:int(n)]

    op_max = op_t.max(dim=-1).values
    op_mean = op_t.mean(dim=-1)
    visibles = (op_max > 0.1).nonzero(as_tuple=False).squeeze(-1)

    if visibles.numel() == 0:
        visibles = torch.arange(N)

    if modo == "top":
        orden = torch.argsort(op_mean[visibles], descending=True)
        return visibles[orden[:n]]

    if modo == "aleatorio":
        g = torch.Generator().manual_seed(int(semilla))
        perm = torch.randperm(len(visibles), generator=g)
        return visibles[perm[:n]]

    if modo == "region":
        if bbox is None:
            raise ValueError("modo region requiere --bbox y0 x0 y1 x1")
        y0, x0, y1, x1 = bbox
        mu0 = mu_t[:, :, 0]
        dentro = (
            (mu0[:, 0] >= y0) & (mu0[:, 0] <= y1) &
            (mu0[:, 1] >= x0) & (mu0[:, 1] <= x1)
        )
        cand = (dentro & (op_max > 0.1)).nonzero(as_tuple=False).squeeze(-1)
        if cand.numel() == 0:
            raise RuntimeError("No encontre gaussianas visibles dentro del bbox.")
        orden = torch.argsort(op_mean[cand], descending=True)
        return cand[orden[:n]]

    raise ValueError(f"modo desconocido: {modo}")


def _png_estatico(mu_t, color_t, idx, fondo, ruta, frame_fondo=0):
    mu_np = mu_t.cpu().numpy()
    col_np = color_t.cpu().numpy()
    idx_np = idx.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 12 * fondo.shape[0] / fondo.shape[1]), dpi=120)
    ax.imshow(np.clip(fondo, 0, 1))

    for k, i in enumerate(idx_np):
        # Color promedio de la gaussiana en el tiempo, para que el marcador tenga sentido.
        col = tuple(np.clip(col_np[i].mean(axis=1), 0, 1))
        ax.plot(mu_np[i, 1, :], mu_np[i, 0, :], color=col, linewidth=1.8, alpha=0.95)
        y0, x0 = mu_np[i, 0, frame_fondo], mu_np[i, 1, frame_fondo]
        ax.add_patch(Circle((x0, y0), radius=7, fill=False, edgecolor=col, linewidth=2))
        ax.text(x0 + 9, y0 - 9, str(int(i)), color=col, fontsize=8, weight="bold")

    ax.set_xlim(0, fondo.shape[1])
    ax.set_ylim(fondo.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"Trayectorias de {len(idx_np)} gaussianas marcadas")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  png estatico -> {ruta}", flush=True)


def _gif_animado(mu_t, color_t, idx, config, fondo_dir, H, W, ruta, fps, paso, estela):
    from PIL import Image

    T = mu_t.shape[-1]
    mu_np = mu_t.cpu().numpy()
    col_np = color_t.cpu().numpy()
    idx_np = idx.cpu().numpy()

    frames_out = []
    for j in range(0, T, max(1, int(paso))):
        fondo = cargar_fondo_desde_dir_o_gt(fondo_dir, config, j, H, W)

        fig, ax = plt.subplots(figsize=(9, 9 * H / W), dpi=100)
        ax.imshow(np.clip(fondo, 0, 1))

        for i in idx_np:
            col = tuple(np.clip(col_np[i, :, j], 0, 1))
            j0 = max(0, j - int(estela))
            ax.plot(mu_np[i, 1, j0:j + 1], mu_np[i, 0, j0:j + 1], color=col, linewidth=1.3, alpha=0.75)
            ax.add_patch(Circle((mu_np[i, 1, j], mu_np[i, 0, j]), radius=7, fill=False, edgecolor=col, linewidth=2))
            ax.text(mu_np[i, 1, j] + 9, mu_np[i, 0, j] - 9, str(int(i)), color=col, fontsize=8, weight="bold")

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        ax.set_title(f"frame {j}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames_out.append(Image.fromarray(buf.copy()))
        plt.close(fig)

        if j == 0 or len(frames_out) % 50 == 0:
            print(f"  gif frame {j}/{T-1}", flush=True)

    if frames_out:
        frames_out[0].save(
            ruta,
            save_all=True,
            append_images=frames_out[1:],
            duration=int(1000 / max(1, int(fps))),
            loop=0,
        )
        print(f"  gif animado -> {ruta} ({len(frames_out)} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--modo", choices=["top", "aleatorio", "region"], default="top")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("y0", "x0", "y1", "x1"))
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--frame_fondo", type=int, default=0)
    ap.add_argument("--fondo_dir", default=None)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--estela", type=int, default=15)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--ids_csv", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    salida = Path(args.salida).resolve() if args.salida else (ckpt.parent.parent / "viz_marcadores")
    salida.mkdir(parents=True, exist_ok=True)

    print("=== viz_trayectorias_marcadores ===", flush=True)
    print(f"checkpoint: {ckpt}", flush=True)
    print(f"salida    : {salida}", flush=True)

    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(ckpt, device=args.device)
    H, W = info["H"], info["W"]

    mu_t, op_t, color_t = _trayectorias_opacidad_color(modelo, matrices_base)
    idx = _seleccionar(mu_t, op_t, args.modo, args.n, args.bbox, args.semilla, ids=args.ids, ids_csv=args.ids_csv)

    print(f"seleccionadas={idx.tolist()}", flush=True)

    frame_fondo = max(0, min(int(args.frame_fondo), info["n_frames"] - 1))
    fondo = cargar_fondo_desde_dir_o_gt(args.fondo_dir, config, frame_fondo, H, W)

    nombre = "ids" if (args.ids or args.ids_csv) else args.modo
    _png_estatico(mu_t, color_t, idx, fondo, salida / f"marcadores_{nombre}_estatico.png", frame_fondo=frame_fondo)

    if args.gif:
        _gif_animado(
            mu_t, color_t, idx, config, args.fondo_dir, H, W,
            salida / f"marcadores_{nombre}_animado.gif",
            args.fps,
            args.paso,
            args.estela,
        )

    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
