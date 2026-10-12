from pathlib import Path
import argparse
import imageio.v2 as imageio
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Carpeta donde estan los frame_0000.png")
    parser.add_argument("--salida", required=True, help="Ruta del video .mp4 de salida")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    carpeta_frames = Path(args.frames)
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    archivos = sorted(carpeta_frames.glob("frame_*.png"))

    if not archivos:
        raise FileNotFoundError(f"No se encontraron frames en: {carpeta_frames}")

    print(f"Frames encontrados: {len(archivos)}")
    print(f"FPS: {args.fps}")
    print(f"Salida: {salida}")

    with imageio.get_writer(str(salida), fps=args.fps, codec="libx264", quality=8) as writer:
        for i, ruta in enumerate(archivos):
            img = Image.open(ruta).convert("RGB")
            writer.append_data(np.array(img))

            if (i + 1) % 100 == 0 or i == 0 or i == len(archivos) - 1:
                print(f"Procesado {i + 1}/{len(archivos)}")

    print("Video generado correctamente.")


if __name__ == "__main__":
    main()