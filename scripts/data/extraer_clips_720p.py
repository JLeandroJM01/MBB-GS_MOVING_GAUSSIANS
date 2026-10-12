"""
Extrae frames de data/videos/video.mp4 como PNGs en data/clips/<nombre_clip>/.

Uso tipico (una sola vez antes de la ablacion):
    python scripts/data/extraer_clips_720p.py

Defaults pensados para el estudio de ablacion de loss:
    - video de entrada : data/videos/video.mp4
    - clip de salida   : data/clips/test30s_clips/
    - duracion         : 30 segundos
    - resolucion       : 1280x720 (720p, conserva aspect 16:9)
    - fps              : el del video original

Para cambiar algo, edita las constantes de arriba o pasa flags:
    --inicio_seg     : segundo donde empieza el clip (default 0)
    --duracion_seg   : duracion en segundos (default 30)
    --max_frames     : tope absoluto de frames a extraer (default None)
    --nombre_clip    : nombre de la carpeta dentro de data/clips/
    --fps            : forzar fps de salida (default = fps del video)
    --H, --W         : forzar resolucion (default 720x1280)
    --forzar         : reextrae aunque la carpeta destino ya tenga PNGs

Importante:
    - cv2 lee BGR; convertimos a RGB antes de guardar.
    - El redimensionado usa INTER_AREA si se reduce, INTER_CUBIC si se agranda
      (regla estandar OpenCV).
"""
import argparse
import sys
from pathlib import Path

import cv2


RAIZ = Path(__file__).resolve().parents[2]
RUTA_VIDEO_DEFAULT = RAIZ / "data" / "videos" / "video.mp4"
RUTA_CLIPS_DEFAULT = RAIZ / "data" / "clips"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=str(RUTA_VIDEO_DEFAULT),
                   help=f"ruta al mp4 (default: {RUTA_VIDEO_DEFAULT})")
    p.add_argument("--nombre_clip", default="test30s_clips",
                   help="nombre de la carpeta destino dentro de data/clips/")
    p.add_argument("--inicio_seg", type=float, default=0.0,
                   help="segundo donde empieza el clip")
    p.add_argument("--duracion_seg", type=float, default=30.0,
                   help="duracion del clip en segundos (0 = hasta el final)")
    p.add_argument("--max_frames", type=int, default=None,
                   help="tope absoluto de frames a extraer")
    p.add_argument("--fps", type=float, default=None,
                   help="forzar fps de salida (default = fps del video)")
    p.add_argument("--H", type=int, default=720, help="alto de salida")
    p.add_argument("--W", type=int, default=1280, help="ancho de salida")
    p.add_argument("--forzar", action="store_true",
                   help="re-extrae aunque la carpeta destino ya tenga PNGs")
    return p.parse_args()


def main():
    args = parse_args()

    ruta_video = Path(args.video)
    if not ruta_video.is_file():
        print(f"ERROR: no existe el video: {ruta_video}", file=sys.stderr)
        sys.exit(2)

    carpeta_salida = RUTA_CLIPS_DEFAULT / args.nombre_clip
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    pngs_existentes = sorted(carpeta_salida.glob("frame_*.png"))
    if pngs_existentes and not args.forzar:
        print(
            f"ERROR: ya hay {len(pngs_existentes)} PNG(s) en {carpeta_salida}.\n"
            f"  - usa --forzar para sobrescribir,\n"
            f"  - o elimina la carpeta manualmente,\n"
            f"  - o cambia --nombre_clip.",
            file=sys.stderr,
        )
        sys.exit(3)

    cap = cv2.VideoCapture(str(ruta_video))
    if not cap.isOpened():
        print(f"ERROR: OpenCV no pudo abrir {ruta_video}", file=sys.stderr)
        sys.exit(4)

    fps_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    print("=== extraer_clips_720p ===", flush=True)
    print(f"  video         : {ruta_video}", flush=True)
    print(f"  fps_video     : {fps_video:.3f}", flush=True)
    print(f"  n_frames_video: {n_total}", flush=True)
    print(f"  resolucion_in : {W_in}x{H_in}", flush=True)

    fps_out = float(args.fps) if args.fps else fps_video
    if fps_out <= 0:
        print("ERROR: fps de salida invalido. Pasa --fps explicito.", file=sys.stderr)
        sys.exit(5)

    H_out, W_out = int(args.H), int(args.W)

    # Frame inicial y cuantos frames extraer.
    frame_inicio_video = int(round(args.inicio_seg * fps_video))
    if frame_inicio_video < 0 or frame_inicio_video >= max(1, n_total):
        print(f"ERROR: inicio_seg fuera de rango.", file=sys.stderr)
        sys.exit(6)

    if args.duracion_seg > 0:
        n_objetivo = int(round(args.duracion_seg * fps_out))
    else:
        n_objetivo = 10**9  # hasta agotar

    if args.max_frames is not None:
        n_objetivo = min(n_objetivo, int(args.max_frames))

    # Si fps_out == fps_video, leemos frames consecutivos. Si difiere,
    # resampleamos por timestamp (no usamos cv2 audio; submuestreo simple).
    paso_video = fps_video / fps_out if fps_out > 0 else 1.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio_video)

    print(f"  inicio_seg    : {args.inicio_seg:.3f}  (frame {frame_inicio_video})", flush=True)
    print(f"  duracion_seg  : {args.duracion_seg:.3f}", flush=True)
    print(f"  fps_out       : {fps_out:.3f}  paso_video={paso_video:.3f}", flush=True)
    print(f"  resolucion_out: {W_out}x{H_out}", flush=True)
    print(f"  n_objetivo    : {n_objetivo}", flush=True)
    print(f"  destino       : {carpeta_salida}", flush=True)

    if args.forzar and pngs_existentes:
        print(f"  borrando {len(pngs_existentes)} PNG previos...", flush=True)
        for p in pngs_existentes:
            p.unlink()

    j_out = 0
    j_video = float(frame_inicio_video)
    interp_reducir = cv2.INTER_AREA
    interp_aumentar = cv2.INTER_CUBIC

    while j_out < n_objetivo:
        idx_video = int(round(j_video))
        if idx_video >= n_total:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_video)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break

        if (frame_bgr.shape[0], frame_bgr.shape[1]) != (H_out, W_out):
            interp = interp_reducir if (H_out < frame_bgr.shape[0]) else interp_aumentar
            frame_bgr = cv2.resize(frame_bgr, (W_out, H_out), interpolation=interp)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        ruta_png = carpeta_salida / f"frame_{j_out:04d}.png"
        # cv2.imwrite espera BGR; convertimos de vuelta para guardar correctamente.
        cv2.imwrite(str(ruta_png), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if (j_out == 0) or ((j_out + 1) % 50 == 0):
            print(f"  frame {j_out + 1:4d}/{n_objetivo}  (video idx {idx_video})", flush=True)

        j_out += 1
        j_video += paso_video

    cap.release()
    print(f"listo. {j_out} PNG(s) en {carpeta_salida}", flush=True)


if __name__ == "__main__":
    main()
