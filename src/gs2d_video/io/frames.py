"""
Conversion de frames entre los formatos en que se mantienen y el formato que
consume el rasterizador.

Un clip puede vivir en tres formas segun la memoria disponible (ver las
opciones frames_en_cpu y frames_en_gpu_uint8 del config):

    - CPU uint8    : el clip entero en RAM, minimo consumo de VRAM
    - GPU uint8    : el clip entero en VRAM sin transferencias por frame
    - GPU float32  : el clip entero en VRAM ya normalizado

El rasterizador siempre necesita float32 en [0, 1] en el device del modelo,
asi que la conversion se hace por frame, justo antes de usarlo.
"""
import torch


def frame_a_device(frame, device):
    """
    Un frame (H, W, 3) a float32 en [0, 1] sobre `device`.

    - uint8 (CPU o GPU) -> float32 normalizado
    - float32 ya en el device -> se devuelve tal cual, sin copia
    """
    if frame.device == device and frame.dtype == torch.float32:
        return frame

    if frame.dtype == torch.uint8:
        return frame.to(device=device, non_blocking=True).float().div_(255.0)

    return frame.to(device=device, dtype=torch.float32, non_blocking=True)


def frames_a_device(frames, device):
    """
    El clip completo (T, H, W, 3) a float32 en [0, 1] sobre `device`.

    Materializa todo el clip en el device, asi que solo conviene en
    resoluciones pequenas. Para clips grandes usar frame_a_device por frame.
    """
    if frames.device == device and frames.dtype == torch.float32:
        return frames

    if frames.dtype == torch.uint8:
        return frames.to(device=device, non_blocking=True).float().div_(255.0)

    return frames.to(device=device, dtype=torch.float32, non_blocking=True)


def frames_a_uint8_numpy(frames):
    """
    El clip completo a numpy uint8 (T, H, W, 3), venga en uint8 o en float
    normalizado, este en CPU o en GPU.
    """
    if frames.dtype == torch.uint8:
        return frames.detach().cpu().numpy()

    return (
        frames.detach()
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
