"""
Extraccion de frames desde un archivo mp4 a una secuencia PNG.

El usuario coloca el video en `tesis_2dgs_video/video/` y este modulo se
invoca AUTOMATICAMENTE desde `correr.py` si el config tiene `video_mp4`
definido. NO hay que correrlo a mano.

Internamente usa ffmpeg (requisito: ffmpeg en el PATH).

Filtro aplicado al video:
  1. crop cuadrado centrado al menor lado (crop='min(iw,ih)':'min(iw,ih)')
  2. scale a (W, H) -- por defecto 256x256
  3. fps opcional (None = nativo del video)

Si la carpeta destino ya contiene los frames esperados, se reutiliza como cache.
La extraccion puede forzarse mediante "forzar_extraccion": true.
"""
import os
import shutil
import subprocess



def _frames_existentes(carpeta):
    """Lista frames PNG existentes en `carpeta` (orden alfabetico)."""
    if not os.path.isdir(carpeta):
        return []
    return sorted(f for f in os.listdir(carpeta)
                    if f.startswith("frame_") and f.endswith(".png"))



def _cache_valido(carpeta, n_frames_esperado):
    """
    Devuelve True si la carpeta YA tiene una extraccion compatible:
      - existe la carpeta
      - tiene al menos un PNG con el patron frame_NNNN.png
      - si n_frames_esperado es int, tiene al menos esa cantidad
    """
    pngs = _frames_existentes(carpeta)
    if not pngs:
        return False
    if n_frames_esperado is not None and len(pngs) < n_frames_esperado:
        return False
    return True



def extraer_frames_de_video(ruta_mp4, carpeta_salida,
                             n_frames=None, fps=None,
                             H=256, W=256,
                             forzar=False):
    """
    Args:
        ruta_mp4         : ruta al mp4 (absoluta o relativa al directorio actual)
        carpeta_salida   : donde escribir los PNGs (frame_0000.png, ...)
        n_frames         : limite de frames; None = todos los que produzca ffmpeg
        fps              : muestrear a este fps; None = fps nativo del video
        H, W             : resolucion final del frame, con crop cuadrado centrado
        forzar           : reextraer aunque la carpeta ya tenga frames

    Returns:
        int -- cantidad de frames finales en `carpeta_salida`.

    Raises:
        FileNotFoundError si el mp4 no existe.
        RuntimeError      si ffmpeg falla o no produce ningun frame.
    """
    if not os.path.isfile(ruta_mp4):
        raise FileNotFoundError(f"video no encontrado: {ruta_mp4}")

    # ---- cache hit? -------------------------------------------------------
    if not forzar and _cache_valido(carpeta_salida, n_frames):
        existentes = _frames_existentes(carpeta_salida)
        # si nos piden n_frames y hay mas, recortamos al limite
        if n_frames is not None and len(existentes) > n_frames:
            for f in existentes[n_frames:]:
                os.remove(os.path.join(carpeta_salida, f))
            existentes = existentes[:n_frames]
        print(f"[preparar_video] cache: ya hay {len(existentes)} frames en {carpeta_salida} "
              f"(usa forzar_extraccion=true para reextraer)", flush=True)
        return len(existentes)

    # ---- limpieza y extraccion fresca -------------------------------------
    if os.path.isdir(carpeta_salida):
        shutil.rmtree(carpeta_salida)
    os.makedirs(carpeta_salida, exist_ok=True)

    # construimos el filtro de video:
    #   crop='min(iw,ih)':'min(iw,ih)'   -> recorte cuadrado al menor lado, centrado
    #   scale=W:H                         -> escala a la resolucion final
    #   fps=N                             -> opcional, frames por segundo
    filtros = ["crop='min(iw\\,ih)':'min(iw\\,ih)'", f"scale={W}:{H}"]
    if fps is not None:
        filtros.append(f"fps={fps}")
    vf = ",".join(filtros)

    cmd = ["ffmpeg", "-y", "-i", ruta_mp4, "-vf", vf]
    if n_frames is not None:
        cmd += ["-frames:v", str(n_frames)]
    cmd += ["-q:v", "2", os.path.join(carpeta_salida, "frame_%04d.png")]

    print(f"[preparar_video] extrayendo frames de {ruta_mp4} a {carpeta_salida}", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)

    resultado = subprocess.run(cmd, capture_output=True, text=True)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"ffmpeg fallo (returncode={resultado.returncode}):\n"
            f"--- stderr ---\n{resultado.stderr}"
        )

    # ffmpeg numera desde 1 (frame_0001.png). Renombramos a 0-based para que
    # los scripts de carga (que ordenan alfabeticamente) lo acepten igual.
    pngs = _frames_existentes(carpeta_salida)
    if not pngs:
        raise RuntimeError("ffmpeg termino OK pero no escribio ningun PNG")
    for i, nombre_viejo in enumerate(pngs):
        nuevo = f"frame_{i:04d}.png"
        if nombre_viejo != nuevo:
            os.rename(os.path.join(carpeta_salida, nombre_viejo),
                      os.path.join(carpeta_salida, nuevo))

    final = _frames_existentes(carpeta_salida)
    print(f"[preparar_video] listo: {len(final)} frames a resolucion {H}x{W}", flush=True)
    return len(final)
