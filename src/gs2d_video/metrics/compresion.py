"""
Metricas de compresion: tamano del modelo, tamano AVIF por frame, ratios.

AVIF (AV1 Image File Format):
-----------------------------
Formato moderno basado en el codec AV1. Comprime muy bien, especialmente
imagenes con contenido suave (gradientes, fondos). Para Python:
    pip install pillow-avif-plugin

Una vez instalado, basta con:
    import pillow_avif    # registra el plugin
    from PIL import Image
    img.save("frame.avif", format="AVIF", quality=80)

Por defecto se evaluan las calidades AVIF 80 y 95.
La lista puede modificarse mediante `calidades_avif`.
"""
import io
import os

import numpy as np
import torch

try:
    import pillow_avif         # noqa: F401  -- registra el plugin
    _AVIF_OK = True
except Exception as _e:
    _AVIF_OK = False
    _AVIF_ERROR = str(_e)

from PIL import Image





# ============================================================================
# tamano del modelo
# ============================================================================

def tamano_modelo_bytes(modelo):
    """
    Cuenta los bytes de los coeficientes del modelo (sd_coefs).
    Incluye metadata minima (grados, N, n_frames, base) -- despreciable
    pero la sumamos para fair reporting.
    """
    sd = modelo.state_dict_coefs()
    total = 0
    for k, v in sd.items():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
        elif isinstance(v, dict):
            # 'grados' es un dict {nombre: int} -- ~6 ints + 6 strings ~ 100 bytes
            total += 100
        elif isinstance(v, str):
            total += len(v)
        elif isinstance(v, int):
            total += 8
    return total



# ============================================================================
# tamano AVIF
# ============================================================================

def _frame_a_pil(frame_array):
    """Convierte (H, W, 3) en [0, 1] o uint8 a PIL.Image RGB."""
    if frame_array.dtype != np.uint8:
        frame_array = (np.clip(frame_array, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(frame_array, mode='RGB')


def _guardar_avif_a_bytes(pil_img, calidad):
    """Guarda la imagen como AVIF y devuelve los bytes (sin escribir disco)."""
    if not _AVIF_OK:
        raise RuntimeError(
            f"pillow_avif no disponible: {_AVIF_ERROR}. "
            f"Instala con 'pip install pillow-avif-plugin'"
        )
    buf = io.BytesIO()
    pil_img.save(buf, format="AVIF", quality=calidad)
    return buf.getvalue()



def tamano_frames_avif(frames_array, calidad=80, carpeta_salvar=None):
    """
    Args:
        frames_array   : numpy (n_frames, H, W, 3) en [0, 1] o uint8
        calidad        : int 0-100 (calidad AVIF)
        carpeta_salvar : si se da, guarda cada frame en disco como
                         <carpeta_salvar>/frame_NNNN.avif

    Returns dict con:
        bytes_por_frame: list[int]
        total_bytes:     int
        promedio_bytes:  float
        calidad:         int
    """
    if isinstance(frames_array, torch.Tensor):
        frames_array = frames_array.detach().cpu().numpy()

    if carpeta_salvar is not None:
        os.makedirs(carpeta_salvar, exist_ok=True)

    bytes_por_frame = []
    for j, fr in enumerate(frames_array):
        pil = _frame_a_pil(fr)
        data = _guardar_avif_a_bytes(pil, calidad)
        bytes_por_frame.append(len(data))
        if carpeta_salvar is not None:
            with open(os.path.join(carpeta_salvar, f"frame_{j:04d}.avif"), "wb") as f:
                f.write(data)

    return {
        'bytes_por_frame': bytes_por_frame,
        'total_bytes':     int(sum(bytes_por_frame)),
        'promedio_bytes':  float(np.mean(bytes_por_frame)),
        'calidad':         calidad,
    }



# ============================================================================
# tamano del video original
# ============================================================================

def tamano_video_original(ruta):
    """
    Si ruta es un archivo: devuelve os.path.getsize.
    Si ruta es una carpeta de PNGs: suma de tamanos.
    """
    if os.path.isfile(ruta):
        return os.path.getsize(ruta)
    if os.path.isdir(ruta):
        total = 0
        for nombre in sorted(os.listdir(ruta)):
            if nombre.startswith("frame_") and nombre.endswith(".png"):
                total += os.path.getsize(os.path.join(ruta, nombre))
        return total
    raise FileNotFoundError(f"no encontrado: {ruta}")



# ============================================================================
# reporte completo
# ============================================================================

def reporte_compresion(modelo, render_batch_array, frames_originales_array,
                        ruta_video_original,
                        calidades_avif=(80, 95),
                        carpeta_avif_originales=None,
                        carpeta_avif_rasterizados=None):
    """
    Args:
        modelo                     : GaussianasPolinomial2D
        render_batch_array         : numpy (n_frames, H, W, 3) en [0, 1] o uint8
        frames_originales_array    : numpy (n_frames, H, W, 3) en [0, 1] o uint8
        ruta_video_original        : ruta a archivo o carpeta PNGs
        calidades_avif             : tupla de int, calidades a evaluar
        carpeta_avif_*             : opcional, guardar los AVIF generados a disco

    Returns dict con todos los tamanos y ratios.

    Si pillow_avif no esta instalado, los tamanos AVIF se reportan como None
    en lugar de hacer crashear el reporte. El resto (tamano del modelo, video
    original, ratio vs original) si se computa siempre.
    """
    bytes_modelo = tamano_modelo_bytes(modelo)
    bytes_video_original = tamano_video_original(ruta_video_original)

    avif_originales_por_calidad = {}
    avif_rasterizados_por_calidad = {}

    if _AVIF_OK:
        for q in calidades_avif:
            ca_o = (carpeta_avif_originales and f"{carpeta_avif_originales}_q{q}")
            ca_r = (carpeta_avif_rasterizados and f"{carpeta_avif_rasterizados}_q{q}")
            avif_originales_por_calidad[q]   = tamano_frames_avif(frames_originales_array,
                                                                   calidad=q, carpeta_salvar=ca_o)
            avif_rasterizados_por_calidad[q] = tamano_frames_avif(render_batch_array,
                                                                   calidad=q, carpeta_salvar=ca_r)
    else:
        print(f"[metricas_compresion] pillow_avif no disponible ({_AVIF_ERROR}); "
              "los tamanos AVIF se reportan como None.")

    reporte = {
        'tamano_modelo_bytes': bytes_modelo,
        'tamano_modelo_kb':    bytes_modelo / 1024.0,
        'tamano_modelo_mb':    bytes_modelo / 1024.0 / 1024.0,
        'tamano_video_original_bytes': bytes_video_original,
        'avif_disponible': _AVIF_OK,
        'avif_originales_por_calidad':   avif_originales_por_calidad,
        'avif_rasterizados_por_calidad': avif_rasterizados_por_calidad,
        'ratio_compresion_vs_original': bytes_video_original / bytes_modelo if bytes_modelo else float('inf'),
    }
    if _AVIF_OK:
        for q in calidades_avif:
            reporte[f'ratio_compresion_vs_avif{q}_total'] = (
                avif_originales_por_calidad[q]['total_bytes'] / bytes_modelo
                if bytes_modelo else float('inf')
            )

    return reporte
