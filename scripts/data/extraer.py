
from pathlib import Path

import cv2


# ============================================================
# CONFIGURACION
# ============================================================

# Nombre del video dentro de:
# tesis_2dgs_video/clips/videos/
VIDEO_NOMBRE = "video.mp4"

# Cuantos frames por segundo quieres extraer
FPS_SALIDA = 30

# Desde que segundo empieza
SEGUNDO_INICIO = 5

# Cuantos segundos quieres extraer
DURACION_SEGUNDOS = 60

# Opciones de resolucion
# Si FORZAR_CUADRADO = False:
#   se mantiene la proporcion y se usa ALTO_SALIDA.
#   Ejemplo: 1080p -> alto 480 => aprox 854x480
#
# Si FORZAR_CUADRADO = True:
#   recorta el centro y redimensiona a TAMANO_CUADRADO x TAMANO_CUADRADO.
FORZAR_CUADRADO = False
TAMANO_CUADRADO = 256*5

# Solo se usa si FORZAR_CUADRADO = False
ALTO_SALIDA = 144*5

# Si True, borra frames viejos de la carpeta de salida antes de generar nuevos
LIMPIAR_SALIDA = True


# ============================================================
# RUTAS RELATIVAS AL PROYECTO
# ============================================================

def obtener_raiz_proyecto():
    """
    Este archivo esta en:
        tesis_2dgs_video/tools/extraer_frames_video.py

    Entonces la raiz del proyecto es:
        tesis_2dgs_video/
    """
    ruta_script = Path(__file__).resolve()
    return ruta_script.parents[2]


def obtener_rutas(video_nombre):
    raiz = obtener_raiz_proyecto()

    video_path = raiz / "data" / "videos" / video_nombre

    nombre_sin_extension = Path(video_nombre).stem
    clip_name = f"{nombre_sin_extension}_clips"

    carpeta_salida = raiz / "data" / "clips" / clip_name

    return raiz, video_path, carpeta_salida, clip_name


# ============================================================
# PROCESAMIENTO
# ============================================================

def limpiar_carpeta_frames(carpeta_salida):
    if not carpeta_salida.exists():
        return

    for archivo in carpeta_salida.iterdir():
        if archivo.is_file() and archivo.name.startswith("frame_") and archivo.suffix.lower() == ".png":
            archivo.unlink()


def redimensionar_manteniendo_proporcion(frame, alto_salida):
    h, w = frame.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError("Frame invalido: dimensiones no validas")

    escala = alto_salida / h
    nuevo_w = int(w * escala)
    nuevo_h = alto_salida

    return cv2.resize(frame, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)


def recortar_centro_cuadrado(frame):
    h, w = frame.shape[:2]
    lado = min(h, w)

    y0 = (h - lado) // 2
    x0 = (w - lado) // 2

    return frame[y0:y0 + lado, x0:x0 + lado]


def procesar_frame(frame):
    if FORZAR_CUADRADO:
        frame = recortar_centro_cuadrado(frame)
        frame = cv2.resize(
            frame,
            (TAMANO_CUADRADO, TAMANO_CUADRADO),
            interpolation=cv2.INTER_AREA
        )
    else:
        frame = redimensionar_manteniendo_proporcion(frame, ALTO_SALIDA)

    return frame


def extraer_frames():
    raiz, video_path, carpeta_salida, clip_name = obtener_rutas(VIDEO_NOMBRE)

    if not video_path.exists():
        raise FileNotFoundError(
            "No se encontro el video.\n"
            f"Ruta esperada:\n{video_path}\n\n"
            "Coloca el video dentro de:\n"
            f"{raiz / 'data' / 'videos'}"
        )

    carpeta_salida.mkdir(parents=True, exist_ok=True)

    if LIMPIAR_SALIDA:
        limpiar_carpeta_frames(carpeta_salida)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps_original <= 0:
        raise RuntimeError("No se pudo obtener el FPS original del video.")

    duracion_total = total_frames / fps_original

    cantidad_frames_salida = int(FPS_SALIDA * DURACION_SEGUNDOS)

    print("=== Extraccion de frames ===")
    print("Raiz proyecto:", raiz)
    print("Video entrada:", video_path)
    print("Clip salida:", clip_name)
    print("Carpeta salida:", carpeta_salida)
    print("FPS original:", fps_original)
    print("Frames totales:", total_frames)
    print("Duracion total:", round(duracion_total, 2), "segundos")
    print("FPS salida:", FPS_SALIDA)
    print("Segundo inicio:", SEGUNDO_INICIO)
    print("Duracion extraida:", DURACION_SEGUNDOS, "segundos")

    if FORZAR_CUADRADO:
        print("Resolucion salida:", f"{TAMANO_CUADRADO}x{TAMANO_CUADRADO}")
    else:
        print("Alto salida:", ALTO_SALIDA, "(ancho proporcional)")

    contador = 0

    for i in range(cantidad_frames_salida):
        tiempo_seg = SEGUNDO_INICIO + (i / FPS_SALIDA)

        if tiempo_seg > duracion_total:
            break

        cap.set(cv2.CAP_PROP_POS_MSEC, tiempo_seg * 1000.0)

        ok, frame = cap.read()

        if not ok:
            print("No se pudo leer frame en segundo:", tiempo_seg)
            continue

        frame = procesar_frame(frame)

        ruta_frame = carpeta_salida / f"frame_{contador:04d}.png"
        cv2.imwrite(str(ruta_frame), frame)

        contador += 1

    cap.release()

    print()
    print("Frames guardados:", contador)
    print("Listo.")
    print()
    print("Ahora en config.json usa:")
    print(f'"clip": "{clip_name}"')
    print(f'"max_frames": {contador}')


if __name__ == "__main__":
    extraer_frames()