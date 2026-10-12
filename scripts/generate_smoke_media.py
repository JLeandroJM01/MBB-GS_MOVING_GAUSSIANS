from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "videos" / "smoke_input.mp4"


def main():
    parser = argparse.ArgumentParser(
        description="Genera el MP4 sintetico utilizado por el smoke test de MBB-GS."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta del MP4 de salida.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Sobrescribe el archivo si ya existe.",
    )
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ERROR: ffmpeg no se encuentra en PATH.")
        return 1

    salida = args.output
    if not salida.is_absolute():
        salida = ROOT / salida

    salida = salida.resolve()
    salida.parent.mkdir(parents=True, exist_ok=True)

    if salida.exists() and not args.force:
        print(f"Smoke input existente: {salida}")
        return 0

    comando = [
        ffmpeg,
        "-y" if args.force else "-n",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=256x144:rate=10",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=2",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        "2",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-ar",
        "16000",
        "-ac",
        "2",
        "-shortest",
        str(salida),
    ]

    print("Generando smoke input...")
    print(salida)

    proceso = subprocess.run(comando, check=False)
    if proceso.returncode != 0:
        print("ERROR: FFmpeg no pudo generar el smoke input.")
        return proceso.returncode

    print("Smoke input generado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
