import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gs2d_video.io.checkpoints import (
    cargar_modelo_desde_checkpoint,
    resolver_base_temporal,
)
from gs2d_video.render.renderer import render_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--fps_origen", type=float, default=30.0)
    parser.add_argument("--fps_salida", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forzar", action="store_true")
    args = parser.parse_args()

    ruta_checkpoint = Path(args.checkpoint)
    carpeta_salida = Path(args.salida)

    if not ruta_checkpoint.is_file():
        raise FileNotFoundError(f"No existe: {ruta_checkpoint}")

    modelo, config, _, info = cargar_modelo_desde_checkpoint(
        ruta_checkpoint,
        device=args.device,
    )

    device = modelo.mu_a0.device
    grados = info["grados"]
    n_gaussianas = int(info["N"])
    n_frames_origen = int(info["n_frames"])
    H = int(info["H"])
    W = int(info["W"])

    base_temporal, construir_matriz_base = resolver_base_temporal(config)

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    existentes = sorted(carpeta_salida.glob("frame_*.png"))
    if existentes and not args.forzar:
        raise RuntimeError(
            f"Ya existen {len(existentes)} frames en {carpeta_salida}. "
            "Usa --forzar para reemplazarlos."
        )

    if args.forzar:
        for archivo in existentes:
            archivo.unlink()

    n_frames_salida = int(
        round((n_frames_origen - 1) * args.fps_salida / args.fps_origen)
    ) + 1
    n_frames_salida = max(2, n_frames_salida)

    print("=== regeneración temporal interpolada ===")
    print(f"checkpoint       : {ruta_checkpoint}")
    print(f"gaussianas       : {n_gaussianas}")
    print(f"resolución       : {H}x{W}")
    print(f"frames originales: {n_frames_origen}")
    print(f"fps original     : {args.fps_origen}")
    print(f"fps salida       : {args.fps_salida}")
    print(f"frames salida    : {n_frames_salida}")
    print(f"base temporal    : {base_temporal}")
    print(f"destino          : {carpeta_salida}")

    grados_unicos = sorted(set(grados.values()))
    matrices_base = {
        grado: construir_matriz_base(
            n_frames=n_frames_salida,
            grado_max=grado,
            device=device,
            dtype=torch.float32,
        )
        for grado in grados_unicos
    }

    with torch.no_grad():
        for j in range(n_frames_salida):
            params = modelo.evaluar_en_frame(j, matrices_base)
            render = render_frame(params, H, W, config).clamp(0, 1)

            imagen = (
                render.mul(255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

            Image.fromarray(imagen).save(
                carpeta_salida / f"frame_{j:04d}.png"
            )

            if j == 0 or (j + 1) % 25 == 0 or j == n_frames_salida - 1:
                print(f"frame {j + 1}/{n_frames_salida}", flush=True)

            del params, render

    print(f"Listo: {n_frames_salida} frames generados.")


if __name__ == "__main__":
    main()
