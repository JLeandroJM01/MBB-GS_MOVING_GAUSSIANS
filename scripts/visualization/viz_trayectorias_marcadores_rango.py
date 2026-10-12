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
def evaluar(a0, hi, B):
    return torch.cat([a0, hi], dim=-1) @ B.T


@torch.no_grad()
def trayectorias_opacidad_color(modelo, matrices_base):
    mu_t = evaluar(modelo.mu_a0, modelo.mu_high, matrices_base[modelo.grados["mu"]])
    op_t = torch.sigmoid(evaluar(modelo.opacity_a0, modelo.opacity_high, matrices_base[modelo.grados["opacity"]]).squeeze(1))
    color_t = torch.sigmoid(evaluar(modelo.color_a0, modelo.color_high, matrices_base[modelo.grados["color"]]))
    return mu_t, op_t, color_t


def leer_ids_csv(path, n):
    ids = []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ids.append(int(row["id"]))
            if len(ids) >= int(n):
                break
    return torch.tensor(ids, dtype=torch.long)


def png_estatico_rango(mu_t, color_t, idx, fondo, ruta, inicio, fin, frame_fondo):
    mu_np = mu_t.cpu().numpy()
    col_np = color_t.cpu().numpy()
    idx_np = idx.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 12 * fondo.shape[0] / fondo.shape[1]), dpi=140)
    ax.imshow(np.clip(fondo, 0, 1))

    for i in idx_np:
        col = tuple(np.clip(col_np[i, :, inicio:fin + 1].mean(axis=1), 0, 1))

        ax.plot(
            mu_np[i, 1, inicio:fin + 1],
            mu_np[i, 0, inicio:fin + 1],
            color=col,
            linewidth=2.0,
            alpha=0.95
        )

        y0, x0 = mu_np[i, 0, frame_fondo], mu_np[i, 1, frame_fondo]
        ax.add_patch(Circle((x0, y0), radius=7, fill=False, edgecolor=col, linewidth=2))
        ax.text(x0 + 9, y0 - 9, str(int(i)), color=col, fontsize=8, weight="bold")

    ax.set_xlim(0, fondo.shape[1])
    ax.set_ylim(fondo.shape[0], 0)
    ax.axis("off")
    ax.set_title(f"Trayectorias de gaussianas seleccionadas | frames {inicio}-{fin}", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    print(f"png -> {ruta}", flush=True)


def gif_animado_rango(mu_t, color_t, idx, config, fondo_dir, H, W, ruta, inicio, fin, fps, paso, estela):
    from PIL import Image

    mu_np = mu_t.cpu().numpy()
    col_np = color_t.cpu().numpy()
    idx_np = idx.cpu().numpy()

    frames_out = []
    paso = max(1, int(paso))

    for j in range(inicio, fin + 1, paso):
        fondo = cargar_fondo_desde_dir_o_gt(fondo_dir, config, j, H, W)

        fig, ax = plt.subplots(figsize=(9, 9 * H / W), dpi=110)
        ax.imshow(np.clip(fondo, 0, 1))

        for i in idx_np:
            col = tuple(np.clip(col_np[i, :, j], 0, 1))
            j0 = max(inicio, j - int(estela))

            ax.plot(
                mu_np[i, 1, j0:j + 1],
                mu_np[i, 0, j0:j + 1],
                color=col,
                linewidth=1.5,
                alpha=0.8
            )

            ax.add_patch(Circle((mu_np[i, 1, j], mu_np[i, 0, j]), radius=7, fill=False, edgecolor=col, linewidth=2))
            ax.text(mu_np[i, 1, j] + 9, mu_np[i, 0, j] - 9, str(int(i)), color=col, fontsize=8, weight="bold")

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        ax.set_title(f"frame {j} | rango {inicio}-{fin}", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()

        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames_out.append(Image.fromarray(buf.copy()))
        plt.close(fig)

        if j == inicio or len(frames_out) % 25 == 0:
            print(f"gif frame {j}/{fin}", flush=True)

    if frames_out:
        frames_out[0].save(
            ruta,
            save_all=True,
            append_images=frames_out[1:],
            duration=int(1000 / max(1, int(fps))),
            loop=0,
        )
        print(f"gif -> {ruta} ({len(frames_out)} frames)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ids_csv", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--inicio", type=int, required=True)
    ap.add_argument("--fin", type=int, required=True)
    ap.add_argument("--frame_fondo", type=int, default=None)
    ap.add_argument("--fondo_dir", default=None)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--paso", type=int, default=2)
    ap.add_argument("--estela", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    salida = Path(args.salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.checkpoint).resolve()

    print("=== viz_trayectorias_marcadores_rango ===", flush=True)
    print(f"checkpoint : {ckpt}", flush=True)
    print(f"salida     : {salida}", flush=True)
    print(f"rango      : {args.inicio}-{args.fin}", flush=True)

    modelo, config, matrices_base, info = cargar_modelo_desde_checkpoint(ckpt, device=args.device)
    H, W = int(info["H"]), int(info["W"])
    T = int(info["n_frames"])

    inicio = max(0, int(args.inicio))
    fin = min(int(args.fin), T - 1)

    if inicio > fin:
        raise RuntimeError(f"Rango invalido: inicio={inicio}, fin={fin}, T={T}")

    frame_fondo = args.frame_fondo
    if frame_fondo is None:
        frame_fondo = (inicio + fin) // 2
    frame_fondo = max(inicio, min(int(frame_fondo), fin))

    mu_t, op_t, color_t = trayectorias_opacidad_color(modelo, matrices_base)

    idx = leer_ids_csv(args.ids_csv, args.n)
    idx = idx[(idx >= 0) & (idx < mu_t.shape[0])]

    print(f"seleccionadas={idx.tolist()}", flush=True)

    fondo = cargar_fondo_desde_dir_o_gt(args.fondo_dir, config, frame_fondo, H, W)

    png_estatico_rango(
        mu_t, color_t, idx, fondo,
        salida / "trayectorias_rango_estatico.png",
        inicio, fin, frame_fondo
    )

    if args.gif:
        gif_animado_rango(
            mu_t, color_t, idx, config, args.fondo_dir, H, W,
            salida / "trayectorias_rango_animado.gif",
            inicio, fin, args.fps, args.paso, args.estela
        )

    print("=== listo ===", flush=True)


if __name__ == "__main__":
    main()
