"""
Visualiza artefactos evolutivos si tu entrenamiento genero:
    outputs/<exp>/verificacion/epochXXXX_frameYYYY.png
    outputs/<exp>/evol_mu/epochXXXX.npz

Uso:
    python scripts/visualization/viz_tira_evolucion.py --exp outputs/EXP --gif

Este script NO crea evol_mu. Solo lo visualiza si existe.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


_RE_VERIF = re.compile(r"epoch(\d+)_frame(\d+)\.png$")
_RE_EVOL = re.compile(r"epoch(\d+)\.npz$")


def _leer_verificacion(carpeta_verif):
    por_frame = defaultdict(list)
    for p in sorted(carpeta_verif.glob("epoch*_frame*.png")):
        m = _RE_VERIF.search(p.name)
        if not m:
            continue
        epoch = int(m.group(1))
        frame = int(m.group(2))
        por_frame[frame].append((epoch, p))
    for frame in por_frame:
        por_frame[frame].sort(key=lambda t: t[0])
    return por_frame


def _grid_convergencia(epoch_pngs, ruta_salida, titulo):
    n = len(epoch_pngs)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), dpi=120)
    if n == 1:
        axes = [axes]
    for ax, (epoch, ruta) in zip(axes, epoch_pngs):
        ax.imshow(np.asarray(Image.open(ruta).convert("RGB")))
        ax.set_title(f"epoch {epoch}", fontsize=10)
        ax.axis("off")
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)
    print(f"  grid -> {ruta_salida}", flush=True)


def _gif_convergencia(epoch_pngs, ruta_salida, fps=2):
    frames = [Image.open(ruta).convert("RGB") for _, ruta in epoch_pngs]
    if not frames:
        return
    dur_ms = int(1000 / max(1, int(fps)))
    frames[0].save(
        ruta_salida,
        save_all=True,
        append_images=frames[1:],
        duration=dur_ms,
        loop=0,
    )
    print(f"  gif  -> {ruta_salida}", flush=True)


def _plot_evolucion_mu(carpeta_evol, ruta_salida, max_gauss=12):
    npzs = []
    for p in carpeta_evol.glob("epoch*.npz"):
        m = _RE_EVOL.search(p.name)
        if m:
            npzs.append((int(m.group(1)), p))
    npzs.sort(key=lambda x: x[0])

    if not npzs:
        print("  (no hay evol_mu/*.npz, salto plot de evolucion de mu)", flush=True)
        return

    datos = []
    for epoch, p in npzs:
        d = np.load(p)
        datos.append((epoch, d["mu"], d["idx"]))

    k_total = datos[0][1].shape[0]
    k = min(int(max_gauss), k_total)
    n_epochs = len(datos)
    cmap = plt.get_cmap("viridis")

    ncol = 4
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), dpi=110)
    axes = np.atleast_1d(axes).flatten()

    for gi in range(k):
        ax = axes[gi]
        for ei, (epoch, mu, idx) in enumerate(datos):
            col = cmap(ei / max(1, n_epochs - 1))
            fila = mu[gi, 0, :]
            columna = mu[gi, 1, :]
            ax.plot(columna, fila, color=col, alpha=0.85, linewidth=1.2, label=f"ep{epoch}" if gi == 0 else None)
        ax.set_title(f"gauss {int(datos[-1][2][gi])}", fontsize=9)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)

    for gi in range(k, len(axes)):
        axes[gi].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(n_epochs, 8), fontsize=8)
    fig.suptitle("Evolucion de trayectoria mu_i(t) por epoch", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(ruta_salida)
    plt.close(fig)
    print(f"  evol_mu -> {ruta_salida}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--salida", default=None)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--max_gauss", type=int, default=12)
    args = ap.parse_args()

    exp = Path(args.exp).resolve()
    salida = Path(args.salida).resolve() if args.salida else (exp / "viz_evolucion")
    salida.mkdir(parents=True, exist_ok=True)

    carpeta_verif = exp / "verificacion"
    carpeta_evol = exp / "evol_mu"

    print("=== viz_tira_evolucion ===", flush=True)
    print(f"exp    : {exp}", flush=True)
    print(f"salida : {salida}", flush=True)

    if carpeta_verif.is_dir():
        por_frame = _leer_verificacion(carpeta_verif)
        if not por_frame:
            print(f"  AVISO: {carpeta_verif} existe pero no encontre epoch*_frame*.png", flush=True)
        for frame, epoch_pngs in sorted(por_frame.items()):
            titulo = f"Convergencia frame {frame} ({len(epoch_pngs)} checkpoints)"
            _grid_convergencia(epoch_pngs, salida / f"convergencia_frame{frame:04d}.png", titulo)
            if args.gif:
                _gif_convergencia(epoch_pngs, salida / f"convergencia_frame{frame:04d}.gif", fps=args.fps)
    else:
        print(f"  AVISO: no existe {carpeta_verif}", flush=True)

    if carpeta_evol.is_dir():
        _plot_evolucion_mu(carpeta_evol, salida / "evolucion_mu_por_epoch.png", max_gauss=args.max_gauss)
    else:
        print(f"  AVISO: no existe {carpeta_evol}", flush=True)

    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
