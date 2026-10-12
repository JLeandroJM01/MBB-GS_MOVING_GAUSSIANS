"""
Extrae un segmento del video original como MP4 con la MISMA duracion y
resolucion que un experimento de Khipu, para poder comparar lado a lado
contra el video reconstruido.

Lee:
  - data/videos/video.mp4 (default)
  - <exp>/info_clip.json para saber n_frames + H + W

Genera:
  - <exp>/video_gt.mp4 con:
      * primeros n_frames del video original (asumiendo fps original)
      * redimensionado a HxW del experimento
      * mismo codec y calidad que video_reconstruido.mp4 generado por
        scripts/data/frames_a_video.py

Uso:
    python scripts/data/extraer_video_gt.py --exp outputs_khipu/fase1_motion
    python scripts/data/extraer_video_gt.py --exp outputs_khipu/fase2_qframe1 --fps 30
    python scripts/data/extraer_video_gt.py --exp outputs_khipu/fase2_qframe8 \\
                                       --video data/videos/video.mp4 \\
                                       --fps 30
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


RAIZ = Path(__file__).resolve().parents[2]
RUTA_VIDEO_DEFAULT = RAIZ / "data" / "videos" / "video.mp4"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True,
                   help="carpeta del experimento (p.ej. outputs_khipu/fase1_motion)")
    p.add_argument("--video", default=str(RUTA_VIDEO_DEFAULT),
                   help=f"ruta al mp4 fuente (default: {RUTA_VIDEO_DEFAULT})")
    p.add_argument("--fps", type=int, default=30,
                   help="fps del video GT a generar. Debe coincidir con el fps "
                        "que usaste al generar video_reconstruido.mp4 (default 30)")
    p.add_argument("--inicio_seg", type=float, default=0.0,
                   help="segundo donde empieza el corte del video original (default 0)")
    p.add_argument("--salida", default=None,
                   help="override del nombre de salida. Por defecto: "
                        "<exp>/video_gt.mp4")
    return p.parse_args()


def main():
    args = parse_args()

    exp = Path(args.exp).resolve()
    if not exp.is_dir():
        print(f"ERROR: no existe la carpeta del experimento: {exp}", file=sys.stderr)
        sys.exit(2)

    info_path = exp / "info_clip.json"
    if not info_path.is_file():
        print(f"ERROR: no existe {info_path}", file=sys.stderr)
        sys.exit(2)

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    n_frames = int(info["n_frames"])
    H = int(info["H"])
    W = int(info["W"])

    ruta_video = Path(args.video)
    if not ruta_video.is_file():
        print(f"ERROR: no existe el video fuente: {ruta_video}", file=sys.stderr)
        sys.exit(3)

    salida = Path(args.salida) if args.salida else (exp / "video_gt.mp4")

    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        print(f"ERROR: cv2 no pudo abrir {ruta_video}", file=sys.stderr)
        sys.exit(4)

    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print("=== extraer_video_gt ===", flush=True)
    print(f"  exp           : {exp}", flush=True)
    print(f"  video fuente  : {ruta_video}", flush=True)
    print(f"  fps_video     : {fps_video:.3f}", flush=True)
    print(f"  n_total_video : {n_total}", flush=True)
    print(f"  resol fuente  : {W_in}x{H_in}", flush=True)
    print(f"  target H x W  : {H} x {W}", flush=True)
    print(f"  target n_frames: {n_frames}", flush=True)
    print(f"  fps_out       : {args.fps}", flush=True)
    print(f"  inicio_seg    : {args.inicio_seg:.3f}", flush=True)
    print(f"  salida        : {salida}", flush=True)

    # Frame inicial.
    fps_out = float(args.fps)
    frame_inicio = int(round(args.inicio_seg * fps_video))
    paso_video = fps_video / fps_out if fps_out > 0 else 1.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    interp_reducir = cv2.INTER_AREA
    interp_aumentar = cv2.INTER_CUBIC

    salida.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(salida), fps=args.fps, codec="libx264", quality=8)

    j_out = 0
    j_video = float(frame_inicio)

    while j_out < n_frames:
        idx_video = int(round(j_video))
        if idx_video >= n_total:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_video)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        if (frame_bgr.shape[0], frame_bgr.shape[1]) != (H, W):
            interp = interp_reducir if (H < frame_bgr.shape[0]) else interp_aumentar
            frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=interp)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

        if (j_out == 0) or ((j_out + 1) % 50 == 0) or (j_out == n_frames - 1):
            print(f"  frame {j_out + 1:4d}/{n_frames}  (video idx {idx_video})", flush=True)

        j_out += 1
        j_video += paso_video

    cap.release()
    writer.close()

    if j_out < n_frames:
        print(f"AVISO: solo se pudieron extraer {j_out}/{n_frames} frames "
              f"(video fuente termino antes)", flush=True)

    print(f"listo. {j_out} frames en {salida}", flush=True)


if __name__ == "__main__":
    main()
