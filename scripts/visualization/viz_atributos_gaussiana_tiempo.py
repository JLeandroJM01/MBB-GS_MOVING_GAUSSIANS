import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint


@torch.no_grad()
def eval_series(a0, hi, B):
    return torch.cat([a0, hi], dim=-1) @ B.T


def leer_primer_id(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            return int(row["id"])
    raise RuntimeError(f"No hay IDs en {path}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--ids_csv", default=None)
    ap.add_argument("--id", type=int, default=None)
    ap.add_argument("--inicio", type=int, default=300)
    ap.add_argument("--fin", type=int, default=600)
    ap.add_argument("--frame_ref", type=int, default=450)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    salida = Path(args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    gid = int(args.id) if args.id is not None else leer_primer_id(args.ids_csv)

    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(args.checkpoint, device=args.device)

    T = int(info["n_frames"])
    H = int(info["H"])
    W = int(info["W"])

    inicio = max(0, int(args.inicio))
    fin = min(int(args.fin), T - 1)
    frame_ref = max(inicio, min(int(args.frame_ref), fin))

    if gid < 0 or gid >= int(info["N"]):
        raise RuntimeError(f"ID fuera de rango: {gid}")

    cols_t = torch.arange(inicio, fin + 1, device=modelo.mu_a0.device)
    frames = np.arange(inicio, fin + 1)

    B_mu = matrices_base[modelo.grados["mu"]][cols_t]
    B_op = matrices_base[modelo.grados["opacity"]][cols_t]
    B_co = matrices_base[modelo.grados["color"]][cols_t]
    B_sc = matrices_base[modelo.grados["scale"]][cols_t]

    log_min = float(np.log(0.5))
    log_max = float(np.log(max(H, W)))

    mu = eval_series(modelo.mu_a0[gid:gid+1], modelo.mu_high[gid:gid+1], B_mu)[0].detach().cpu().numpy()
    op = torch.sigmoid(
        eval_series(modelo.opacity_a0[gid:gid+1], modelo.opacity_high[gid:gid+1], B_op).squeeze(1)
    )[0].detach().cpu().numpy()
    color = torch.sigmoid(
        eval_series(modelo.color_a0[gid:gid+1], modelo.color_high[gid:gid+1], B_co)
    )[0].detach().cpu().numpy()
    scale = torch.exp(
        eval_series(modelo.scale_a0[gid:gid+1], modelo.scale_high[gid:gid+1], B_sc).clamp(min=log_min, max=log_max)
    )[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.4), dpi=150)
    axes = axes.flatten()

    axes[0].plot(frames, mu[1], label="x(t) columna")
    axes[0].plot(frames, mu[0], label="y(t) fila")
    axes[0].axvline(frame_ref, linestyle="--", linewidth=1)
    axes[0].set_title("Posición del centro")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("px")
    axes[0].legend(fontsize=8)

    axes[1].plot(frames, op, label="opacidad")
    axes[1].axvline(frame_ref, linestyle="--", linewidth=1)
    axes[1].set_title("Opacidad temporal")
    axes[1].set_xlabel("frame")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)

    axes[2].plot(frames, scale[1], label="scale x")
    axes[2].plot(frames, scale[0], label="scale y")
    axes[2].axvline(frame_ref, linestyle="--", linewidth=1)
    axes[2].set_title("Escala")
    axes[2].set_xlabel("frame")
    axes[2].set_ylabel("px")
    axes[2].legend(fontsize=8)

    axes[3].plot(frames, color[0], label="R(t)")
    axes[3].plot(frames, color[1], label="G(t)")
    axes[3].plot(frames, color[2], label="B(t)")
    axes[3].axvline(frame_ref, linestyle="--", linewidth=1)
    axes[3].set_title("Color RGB aprendido")
    axes[3].set_xlabel("frame")
    axes[3].set_ylim(0, 1.05)
    axes[3].legend(fontsize=8)

    fig.suptitle(f"Gaussiana {gid}: atributos temporales | frames {inicio}-{fin}", fontsize=13)
    fig.tight_layout()

    out = salida / "atributos_gaussiana_seleccionada.png"
    fig.savefig(out)
    plt.close(fig)

    with open(salida / "gaussiana_id.txt", "w", encoding="utf-8") as f:
        f.write(str(gid))

    print(f"gaussiana_id={gid}")
    print(f"figura={out}")


if __name__ == "__main__":
    main()
