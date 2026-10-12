"""
Extrae frames del video original a la misma resolucion del entrenamiento,
empezando desde un frame especifico del video fuente. Se usa para tener los
frames GT exactamente alineados con los frames renderizados por el modelo,
y poder hacer comparaciones lado a lado.

Uso:
    python scripts/data/extraer_frames_originales.py --inicio_frame 150
    python scripts/data/extraer_frames_originales.py --inicio_frame 150 --n_frames 600
    python scripts/data/extraer_frames_originales.py --inicio_frame 150 --H 720 --W 1280

Por defecto:
    - source : data/videos/video.mp4
    - destino: data/clips/frames_originales/frame_NNNN.png
    - n_frames: 600 (= 20 s a 30 fps, la misma duracion de los renders)
    - resolucion: 1280 x 720 (W x H, la resolucion de tus entrenamientos)

El frame "inicio_frame" se renombra como frame_0000.png en la salida, de modo
que el indice queda alineado con la numeracion del modelo:
    data/clips/frames_originales/frame_0000.png  <-- frame inicio_frame del video original
    data/clips/frames_originales/frame_0001.png  <-- frame inicio_frame + 1
    ...
"""
import argparse
import os
import sys
from pathlib import Path

import cv2


RAIZ = Path(__file__).resolve().parents[2]
RUTA_VIDEO_DEFAULT = RAIZ / "data" / "videos" / "video.mp4"
RUTA_SALIDA_DEFAULT = RAIZ / "data" / "clips" / "frames_originales"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inicio_frame", type=int, required=True,
                   help="indice del frame del video fuente donde EMPEZAR la extraccion. "
                        "Este frame se renombra a frame_0000.png en la salida.")
    p.add_argument("--n_frames", type=int, default=600,
                   help="cuantos frames extraer (default: 600 = 20s a 30fps)")
    p.add_argument("--video", default=str(RUTA_VIDEO_DEFAULT),
                   help=f"ruta al mp4 fuente (default: {RUTA_VIDEO_DEFAULT})")
    p.add_argument("--salida", default=str(RUTA_SALIDA_DEFAULT),
                   help=f"carpeta donde guardar los PNGs (default: {RUTA_SALIDA_DEFAULT})")
    p.add_argument("--H", type=int, default=720, help="alto de salida (default 720)")
    p.add_argument("--W", type=int, default=1280, help="ancho de salida (default 1280)")
    p.add_argument("--forzar", action="store_true",
                   help="re-extrae aunque la carpeta destino ya tenga PNGs")
    return p.parse_args()


def main():
    args = parse_args()

    ruta_video = Path(args.video)
    if not ruta_video.is_file():
        print(f"ERROR: no existe el video fuente: {ruta_video}", file=sys.stderr)
        sys.exit(2)

    carpeta_salida = Path(args.salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    pngs_existentes = sorted(carpeta_salida.glob("frame_*.png"))
    if pngs_existentes and not args.forzar:
        print(
            f"ERROR: ya hay {len(pngs_existentes)} PNG(s) en {carpeta_salida}.\n"
            f"  - usa --forzar para sobrescribir,\n"
            f"  - o elimina la carpeta a mano,\n"
            f"  - o cambia --salida.",
            file=sys.stderr,
        )
        sys.exit(3)

    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        print(f"ERROR: cv2 no pudo abrir {ruta_video}", file=sys.stderr)
        sys.exit(4)

    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    inicio = int(args.inicio_frame)
    n_objetivo = int(args.n_frames)
    H_out, W_out = int(args.H), int(args.W)

    print("=== extraer_frames_originales ===", flush=True)
    print(f"  video fuente   : {ruta_video}", flush=True)
    print(f"  fps_video      : {fps_video:.3f}", flush=True)
    print(f"  n_total_video  : {n_total}", flush=True)
    print(f"  resol fuente   : {W_in} x {H_in}", flush=True)
    print(f"  inicio_frame   : {inicio}  ({inicio / max(fps_video, 1e-9):.3f}s)", flush=True)
    print(f"  n_frames       : {n_objetivo}", flush=True)
    print(f"  resol salida   : {W_out} x {H_out}", flush=True)
    print(f"  destino        : {carpeta_salida}", flush=True)

    if inicio < 0 or inicio >= n_total:
        print(f"ERROR: inicio_frame={inicio} fuera de rango [0, {n_total - 1}]",
              file=sys.stderr)
        sys.exit(5)

    if inicio + n_objetivo > n_total:
        nuevo = n_total - inicio
        print(f"AVISO: solo hay {nuevo} frames a partir de {inicio} "
              f"(pediste {n_objetivo}). Se extraeran {nuevo}.", flush=True)
        n_objetivo = nuevo

    if args.forzar and pngs_existentes:
        print(f"  borrando {len(pngs_existentes)} PNG previos...", flush=True)
        for p in pngs_existentes:
            p.unlink()

    interp_reducir = cv2.INTER_AREA
    interp_aumentar = cv2.INTER_CUBIC

    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)

    j_out = 0
    while j_out < n_objetivo:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            print(f"AVISO: read fallo en frame {inicio + j_out}; deteniendo.",
                  flush=True)
            break

        if (frame_bgr.shape[0], frame_bgr.shape[1]) != (H_out, W_out):
            interp = interp_reducir if (H_out < frame_bgr.shape[0]) else interp_aumentar
            frame_bgr = cv2.resize(frame_bgr, (W_out, H_out), interpolation=interp)

        # cv2.imwrite escribe BGR directamente, que es lo correcto.
        ruta_png = carpeta_salida / f"frame_{j_out:04d}.png"
        cv2.imwrite(str(ruta_png), frame_bgr)

        if (j_out == 0) or ((j_out + 1) % 50 == 0) or (j_out == n_objetivo - 1):
            print(f"  frame {j_out + 1:4d}/{n_objetivo}  "
                  f"(video idx {inicio + j_out})", flush=True)

        j_out += 1

    cap.release()
    print(f"listo. {j_out} PNG(s) en {carpeta_salida}", flush=True)


if __name__ == "__main__":
    main()
