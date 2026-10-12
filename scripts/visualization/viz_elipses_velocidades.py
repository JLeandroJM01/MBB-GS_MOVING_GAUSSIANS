"""
Visualiza elipses y velocidades de gaussianas desde un checkpoint.

Uso:
    python scripts/visualization/viz_elipses_velocidades.py --checkpoint outputs/EXP/checkpoints/checkpoint_final.pt --frame 375 --fondo_dir outputs/EXP/frames_renderizados

Mejora:
    Puedes pasar IDs especificos:
    python scripts/visualization/viz_elipses_velocidades.py --checkpoint ... --frame 375 --ids 123 456 789

    O leer ids desde un CSV generado por viz_gaussian_stats.py:
    python scripts/visualization/viz_elipses_velocidades.py --checkpoint ... --frame 375 --ids_csv outputs/EXP/viz_stats/top_movimiento.csv --n_max 100
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint, cargar_fondo_desde_dir_o_gt


def _leer_ids_csv(path, n_max):
    ids = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if "id" in row:
                ids.append(int(row["id"]))
            if len(ids) >= int(n_max):
                break
    return ids


@torch.no_grad()
def _params_en_frame(modelo, matrices_base, j):
    out = modelo.evaluar_en_frame(j, matrices_base)
    return out["mu"], out["scale"], out["theta"], out["opacity"], out["color"]


def _plot_elipses(mu, scale, theta, opacity, fondo, ruta, indices, k_sigma):
    mu = mu.cpu().numpy()
    scale = scale.cpu().numpy()
    theta = theta.cpu().numpy()
    op = opacity.cpu().numpy()

    H, W = fondo.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 12 * H / W), dpi=120)
    ax.imshow(np.clip(fondo, 0, 1))

    elipses = []
    alphas = []
    for i in indices:
        cy, cx = mu[i, 0], mu[i, 1]
        sy, sx = scale[i, 0], scale[i, 1]
        ang = np.degrees(theta[i])
        e = Ellipse((cx, cy), width=2 * k_sigma * sx, height=2 * k_sigma * sy, angle=ang)
        elipses.append(e)
        alphas.append(float(op[i]))

    pc = PatchCollection(elipses, facecolor="none", edgecolor="cyan", linewidths=0.6, alpha=0.65)
    ax.add_collection(pc)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_title(f"Elipses de {len(indices)} gaussianas (k_sigma={k_sigma})")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  elipses -> {ruta}", flush=True)


def _plot_velocidades(modelo, matrices_base, j, fondo, ruta, indices, escala_flecha):
    n_frames = modelo.n_frames
    j1 = min(j + 1, n_frames - 1)
    mu_j, _, _, op_j, _ = _params_en_frame(modelo, matrices_base, j)
    mu_j1, _, _, _, _ = _params_en_frame(modelo, matrices_base, j1)

    v = (mu_j1 - mu_j).cpu().numpy()
    mu = mu_j.cpu().numpy()

    H, W = fondo.shape[:2]
    idx = np.asarray(indices, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(12, 12 * H / W), dpi=120)
    ax.imshow(np.clip(fondo, 0, 1))
    mag = np.linalg.norm(v[idx], axis=1)

    q = ax.quiver(
        mu[idx, 1], mu[idx, 0],
        v[idx, 1], v[idx, 0],
        mag,
        angles="xy",
        scale_units="xy",
        scale=1.0 / max(1e-6, float(escala_flecha)),
        cmap="plasma",
        width=0.002,
    )
    plt.colorbar(q, ax=ax, fraction=0.025, label="|v| px/frame")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_title(f"Velocidades mu(t+1)-mu(t), frame {j}, {len(indices)} gaussianas")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  velocidades -> {ruta}", flush=True)


def _elegir_indices(opacity, n_max, ids=None, ids_csv=None):
    N = int(opacity.shape[0])
    if ids_csv:
        ids = _leer_ids_csv(ids_csv, n_max)
    if ids:
        return [int(i) for i in ids if 0 <= int(i) < N][:int(n_max)]

    op = opacity.detach().cpu().numpy()
    return np.argsort(-op)[:int(n_max)].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--fondo_dir", default=None)
    ap.add_argument("--n_max", type=int, default=2000)
    ap.add_argument("--k_sigma", type=float, default=2.0)
    ap.add_argument("--escala_flecha", type=float, default=5.0)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--ids_csv", default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint).resolve()
    salida = Path(args.salida).resolve() if args.salida else (ckpt.parent.parent / "viz_elipses_vel")
    salida.mkdir(parents=True, exist_ok=True)

    print("=== viz_elipses_velocidades ===", flush=True)
    print(f"checkpoint: {ckpt}", flush=True)
    print(f"frame     : {args.frame}", flush=True)
    print(f"salida    : {salida}", flush=True)

    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(ckpt, device=args.device)
    H, W = info["H"], info["W"]

    frame = max(0, min(int(args.frame), info["n_frames"] - 1))
    fondo = cargar_fondo_desde_dir_o_gt(args.fondo_dir, config, frame, H, W)
    mu, scale, theta, opacity, color = _params_en_frame(modelo, matrices_base, frame)

    indices = _elegir_indices(opacity, args.n_max, ids=args.ids, ids_csv=args.ids_csv)
    print(f"indices usados: {len(indices)}", flush=True)
    if len(indices) <= 50:
        print("ids:", indices, flush=True)

    _plot_elipses(mu, scale, theta, opacity, fondo, salida / f"elipses_frame{frame:04d}.png", indices, args.k_sigma)
    _plot_velocidades(modelo, matrices_base, frame, fondo, salida / f"velocidades_frame{frame:04d}.png", indices, args.escala_flecha)

    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
