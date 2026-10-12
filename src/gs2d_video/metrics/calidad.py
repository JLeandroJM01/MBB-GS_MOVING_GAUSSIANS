"""
Metricas de calidad por frame y agregadas.

PSNR  : log10 del MSE inverso. data_range = 1.0.
SSIM  : pytorch_msssim si esta, sino implementacion propia (ventana 11).
LPIPS : paquete lpips con backbone alex. Se puede desactivar desde config.

Modo debug recomendado:
    reporte_completo(..., usar_ssim=False, usar_lpips=False)

Modo final recomendado:
    reporte_completo(..., usar_ssim=True, usar_lpips=True)
"""
import numpy as np
import torch
import torch.nn.functional as F


# pytorch-msssim opcional
try:
    from pytorch_msssim import ssim as _ssim_externo
    _USAR_MSSSIM = True
except Exception:
    _USAR_MSSSIM = False


# LPIPS opcional, instanciado bajo demanda
_LPIPS_FN = None
_LPIPS_INTENTADO = False


def _obtener_lpips(device):
    global _LPIPS_FN, _LPIPS_INTENTADO
    if not _LPIPS_INTENTADO:
        _LPIPS_INTENTADO = True
        try:
            import lpips
            _LPIPS_FN = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:
            print(f"[metricas_calidad] lpips no disponible ({e}), se reportara None", flush=True)
            _LPIPS_FN = None
    return _LPIPS_FN


def calcular_psnr(render, gt):
    """render, gt: (H, W, 3) en [0, 1]."""
    mse = torch.mean((render - gt) ** 2)
    if mse <= 0:
        return float("inf")
    return float(-10.0 * torch.log10(mse))


def _kernel_gauss_2d(tamano=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(tamano, dtype=dtype, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    return g1[:, None] * g1[None, :]


def calcular_ssim(render, gt):
    """render, gt: (H, W, 3) en [0, 1]. Devuelve float."""
    x_b = render.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    y_b = gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

    if _USAR_MSSSIM:
        return float(_ssim_externo(x_b, y_b, data_range=1.0, size_average=True, win_size=11))

    C = x_b.shape[1]
    device = x_b.device
    kernel = _kernel_gauss_2d(11, 1.5, device, x_b.dtype).expand(C, 1, 11, 11)

    mu_x = F.conv2d(x_b, kernel, padding=5, groups=C)
    mu_y = F.conv2d(y_b, kernel, padding=5, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sx2 = F.conv2d(x_b * x_b, kernel, padding=5, groups=C) - mu_x2
    sy2 = F.conv2d(y_b * y_b, kernel, padding=5, groups=C) - mu_y2
    sxy = F.conv2d(x_b * y_b, kernel, padding=5, groups=C) - mu_xy

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_xy + C1) * (2 * sxy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2)
    return float((num / den).mean())


def calcular_lpips(render, gt, device=None):
    """render, gt: (H, W, 3) en [0, 1]. Devuelve float o None."""
    if device is None:
        device = render.device

    fn = _obtener_lpips(device)
    if fn is None:
        return None

    x_b = (render.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0)
    y_b = (gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1) * 2.0 - 1.0)

    param_device = next(fn.parameters()).device
    x_b = x_b.to(param_device)
    y_b = y_b.to(param_device)

    with torch.no_grad():
        return float(fn(x_b, y_b).item())



def _agregados(valores, prefijo):
    """Devuelve dict con {prefijo}_promedio/min/max/p5/std. Ignora None."""
    validos = [v for v in valores if v is not None and not np.isinf(v)]
    if not validos:
        return {
            f"{prefijo}_promedio": None,
            f"{prefijo}_min":      None,
            f"{prefijo}_max":      None,
            f"{prefijo}_p5":       None,
            f"{prefijo}_std":      None,
        }
    arr = np.asarray(validos, dtype=np.float64)
    return {
        f"{prefijo}_promedio": float(arr.mean()),
        f"{prefijo}_min":      float(arr.min()),
        f"{prefijo}_max":      float(arr.max()),
        f"{prefijo}_p5":       float(np.percentile(arr, 5.0)),
        f"{prefijo}_std":      float(arr.std()),
    }


def _psnr_temporal_por_frame(render_batch, frames_gt, data_range=2.0):
    """
    PSNR de las diferencias frame-a-frame:
        dR  = R_t  - R_{t-1}        en [-1, 1]
        dGT = GT_t - GT_{t-1}       en [-1, 1]
        err = dR - dGT              en [-2,  2]
        MSE = mean(err^2)
        PSNR_temp_t = 10 * log10(data_range^2 / MSE)

    data_range=2.0 porque la amplitud maxima de las diferencias es 2 (de -1 a 1).
    Devuelve lista de longitud n_frames; el indice 0 es None porque no hay frame previo.
    """
    n = render_batch.shape[0]
    out = [None]

    for t in range(1, n):
        dR = render_batch[t] - render_batch[t - 1]
        dGT = frames_gt[t] - frames_gt[t - 1]
        err = dR - dGT
        mse = float(torch.mean(err * err).item())
        if mse <= 0.0:
            out.append(float("inf"))
        else:
            out.append(10.0 * float(np.log10((data_range ** 2) / mse)))

    return out


@torch.no_grad()
def reporte_completo_streaming(
    n_frames,
    fn_obtener_par,
    device=None,
    usar_ssim=True,
    usar_lpips=False,
    data_range_temporal=2.0,
):
    """
    Calcula el mismo dict que reporte_completo() pero SIN apilar renders.

    Args:
        n_frames        : int.
        fn_obtener_par  : callable j -> (render_j_clamped, gt_j_clamped),
                          ambos (H,W,3) en [0,1], en el mismo device.
                          Se llama exactamente una vez por frame.
        device          : device para LPIPS.
        usar_ssim, usar_lpips : igual que reporte_completo.
        data_range_temporal : usado para el PSNR temporal (default 2.0).

    Para PSNR temporal mantiene 2 frames en memoria (render previo y gt previo).
    El indice 0 de psnr_temporal_por_frame siempre es None.
    """
    psnrs, ssims, lpipss = [], [], []
    psnrs_temp = [None]

    prev_render = None
    prev_gt = None

    for j in range(n_frames):
        render_j, gt_j = fn_obtener_par(j)

        psnrs.append(calcular_psnr(render_j, gt_j))

        if usar_ssim:
            ssims.append(calcular_ssim(render_j, gt_j))
        else:
            ssims.append(None)

        if usar_lpips:
            lpipss.append(calcular_lpips(render_j, gt_j, device=device))
        else:
            lpipss.append(None)

        if j > 0 and prev_render is not None and prev_gt is not None:
            dR = render_j - prev_render
            dGT = gt_j - prev_gt
            err = dR - dGT
            mse = float(torch.mean(err * err).item())
            if mse <= 0.0:
                psnrs_temp.append(float("inf"))
            else:
                psnrs_temp.append(
                    10.0 * float(np.log10((data_range_temporal ** 2) / mse))
                )

        prev_render = render_j
        prev_gt = gt_j

    salida = {
        "psnr_por_frame":          psnrs,
        "ssim_por_frame":          ssims,
        "lpips_por_frame":         lpipss,
        "psnr_temporal_por_frame": psnrs_temp,
    }
    salida.update(_agregados(psnrs,      "psnr"))
    salida.update(_agregados(ssims,      "ssim"))
    salida.update(_agregados(lpipss,     "lpips"))
    salida.update(_agregados(psnrs_temp, "psnr_temporal"))

    return salida


@torch.no_grad()
def reporte_completo(render_batch, frames_gt, device=None, usar_ssim=True, usar_lpips=False):
    """
    Calcula metricas por frame y agregados (promedio, min, max, p5, std).

    Args:
        render_batch : (n_frames, H, W, 3) en [0, 1]
        frames_gt    : (n_frames, H, W, 3) en [0, 1]
        usar_ssim    : si False, no calcula SSIM. Util para debug rapido.
        usar_lpips   : si False, no carga AlexNet/LPIPS. Util para debug rapido.

    Devuelve dict con:
        psnr_por_frame, ssim_por_frame, lpips_por_frame      : list[float|None]
        psnr_temporal_por_frame                              : list[float|None]
                                                               (indice 0 es None)
        psnr_promedio/min/max/p5/std                         : float|None
        ssim_promedio/min/max/p5/std                         : float|None
        lpips_promedio/min/max/p5/std                        : float|None
        psnr_temporal_promedio/min/max/p5/std                : float|None
    """
    n_frames = render_batch.shape[0]
    psnrs, ssims, lpipss = [], [], []

    for j in range(n_frames):
        psnrs.append(calcular_psnr(render_batch[j], frames_gt[j]))

        if usar_ssim:
            ssims.append(calcular_ssim(render_batch[j], frames_gt[j]))
        else:
            ssims.append(None)

        if usar_lpips:
            lpipss.append(calcular_lpips(render_batch[j], frames_gt[j], device=device))
        else:
            lpipss.append(None)

    psnrs_temp = _psnr_temporal_por_frame(render_batch, frames_gt, data_range=2.0)

    salida = {
        "psnr_por_frame":          psnrs,
        "ssim_por_frame":          ssims,
        "lpips_por_frame":         lpipss,
        "psnr_temporal_por_frame": psnrs_temp,
    }
    salida.update(_agregados(psnrs,      "psnr"))
    salida.update(_agregados(ssims,      "ssim"))
    salida.update(_agregados(lpipss,     "lpips"))
    salida.update(_agregados(psnrs_temp, "psnr_temporal"))

    return salida
