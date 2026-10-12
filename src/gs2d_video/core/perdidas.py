"""
Funciones de perdida para GS2D video.

Modos disponibles con config["tipo_loss"]:
  - baseline        : L1 + DSSIM opcional
  - l1_mse          : L1 + MSE + DSSIM opcional
  - motion          : L1 ponderado por movimiento del GT + DSSIM opcional
  - hard            : L1 ponderado por error actual + DSSIM opcional
  - edge            : L1 + Sobel edge loss + DSSIM opcional
  - temporal        : L1 + loss temporal + DSSIM opcional
  - motion_temporal : motion + MSE opcional + temporal opcional + DSSIM opcional
  - combo           : activa solo las lambdas > 0

Knobs universales que afectan TODOS los tipo_loss (orthogonales):
  - exponente_pixel : float (default 1.0). p-norm sobre los errores absolutos de pixel.
                      p=1  -> mean(|R-T|)        (actual, sin cambio)
                      p=2  -> mean(|R-T|^2)      (estilo MSE)
                      p=4  -> mean(|R-T|^4)      (penaliza fuerte pixeles malos)
                      p=10 -> casi max(|R-T|)    (solo el pixel peor pesa)
                      Da gradiente proporcional a |R-T|^(p-1): "hard pixel mining"
                      continuo. Castiga regiones con error grande mucho mas que el promedio.
  - usar_pnorm_root : bool (default false). Si true, devuelve (mean(diff^p))^(1/p)
                      preservando la escala original. Para Adam el efecto es minimo.
  - usar_max_pixel  : bool (default false). Si true, ignora exponente_pixel y usa max
                      puro sobre los pixeles. UNICAMENTE el peor pixel recibe gradiente.
                      Inestable salvo en early training; preferir exponente_pixel alto.

Notas:
- Si una lambda esta en 0, esa parte NO se calcula.
- Este archivo funciona con el raster CUDA actual si usar_loss_cuda=false.
- Para motion real, trainer.py debe pasar frames_all y frame_idx/frame_indices.
- Para temporal real por frame, trainer.py debe pasar prev_render y prev_target.
- exponente_pixel se aplica SOLO al termino L1/diff_abs. MSE/DSSIM/edge/temporal no se tocan.
"""
import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim as _ssim_externo
    _USAR_MSSSIM = True
except Exception:
    _USAR_MSSSIM = False


_EPS = 1e-8


def _kernel_gauss_2d(tamano=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(tamano, dtype=dtype, device=device) - (tamano - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum().clamp_min(_EPS)
    return g1[:, None] * g1[None, :]


def _dssim_propio(x_b, y_b):
    """DSSIM diferenciable. x_b, y_b: (B, C, H, W)."""
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
    ssim_map = ((2 * mu_xy + C1) * (2 * sxy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sx2 + sy2 + C2)
    )
    return (1.0 - ssim_map.mean()) / 2.0


def _config_from_arg(config_or_lambda, lambda_dssim=None):
    """
    Compatibilidad:
    - config: loss_render_frame(render, target, config)
    - compatible: loss_render_frame(render, target, lambda_dssim=0.2)
    - compatible: loss_render_batch(render, target, 0.2)
    """
    if isinstance(config_or_lambda, dict):
        config = dict(config_or_lambda)
    elif config_or_lambda is None:
        config = {"tipo_loss": "baseline", "lambda_dssim": 0.2}
    else:
        config = {"tipo_loss": "baseline", "lambda_dssim": float(config_or_lambda)}

    if lambda_dssim is not None:
        config["lambda_dssim"] = float(lambda_dssim)

    return config


def _get_float(config, key, default=0.0):
    return float(config.get(key, default))


def _loss_dssim_batch(render_batch, frames_gt):
    x_b = render_batch.permute(0, 3, 1, 2).clamp(0, 1)
    y_b = frames_gt.permute(0, 3, 1, 2).clamp(0, 1)

    if _USAR_MSSSIM:
        ssim_val = _ssim_externo(x_b, y_b, data_range=1.0, size_average=True, win_size=11)
        return (1.0 - ssim_val) / 2.0

    return _dssim_propio(x_b, y_b)


def _mezclar_dssim(loss_base, render_batch, frames_gt, config):
    lambda_dssim = _get_float(config, "lambda_dssim", 0.0)
    if lambda_dssim <= 0.0:
        return loss_base

    dssim = _loss_dssim_batch(render_batch, frames_gt)
    return (1.0 - lambda_dssim) * loss_base + lambda_dssim * dssim


def _weighted_mean(diff_abs, weight=None):
    """diff_abs: (B,H,W,3). weight: None o (B,H,W,1)."""
    if weight is None:
        return diff_abs.mean()

    if weight.dim() == 3:
        weight = weight.unsqueeze(-1)

    weight = weight.to(device=diff_abs.device, dtype=diff_abs.dtype)
    weight = weight.expand_as(diff_abs)

    return (diff_abs * weight).sum() / weight.sum().clamp_min(_EPS)


def _aggregate_pixel(diff_abs, weight, config):
    """
    Agregacion p-norm/max de los errores absolutos de pixel dentro de un batch.

    Args:
        diff_abs : (B,H,W,3) o (H,W,3) -- abs(R - T).
        weight   : None o (B,H,W,1)/(B,H,W) -- peso por pixel (motion/hard/etc).
        config   : dict de config (usado para leer exponente_pixel, usar_max_pixel, usar_pnorm_root).

    Returns:
        scalar tensor con el loss agregado.

    Comportamiento:
        - usar_max_pixel=true   -> max(diff_abs * weight). Solo el peor pixel.
        - exponente_pixel == 1  -> equivalente a _weighted_mean(diff_abs, weight).
        - exponente_pixel  > 1  -> mean(diff_abs^p * weight) -- "hard pixel mining" continuo.
                                   Si usar_pnorm_root=true se aplica (.)^(1/p) al resultado.

    Notas:
        - Con max puro solo el peor pixel recibe gradiente.
          Util como experimento puntual, no como entrenamiento estable.
        - Para p alto (>= 8) el comportamiento se acerca asintoticamente al max,
          pero preservando algo de gradiente para los demas pixeles -- mas estable.
    """
    if bool(config.get("usar_max_pixel", False)):
        if weight is None:
            return diff_abs.max()
        if weight.dim() == 3:
            weight = weight.unsqueeze(-1)
        weight = weight.to(device=diff_abs.device, dtype=diff_abs.dtype)
        weight = weight.expand_as(diff_abs)
        return (diff_abs * weight).max()

    exponente = float(config.get("exponente_pixel", 1.0))

    if exponente == 1.0:
        return _weighted_mean(diff_abs, weight)

    # p-norm: aplica power per-pixel, luego promedio (eventualmente ponderado).
    # clamp_min para evitar 0^(p-1) cuando p no es entero (gradiente NaN en x=0).
    base = diff_abs.clamp_min(_EPS) if exponente != int(exponente) else diff_abs
    pow_diff = base ** exponente

    if weight is None:
        result = pow_diff.mean()
    else:
        if weight.dim() == 3:
            weight = weight.unsqueeze(-1)
        weight = weight.to(device=pow_diff.device, dtype=pow_diff.dtype)
        weight = weight.expand_as(pow_diff)
        result = (pow_diff * weight).sum() / weight.sum().clamp_min(_EPS)

    if bool(config.get("usar_pnorm_root", False)):
        result = result.clamp_min(_EPS) ** (1.0 / exponente)

    return result


def _normalizar_indices(frame_indices, device):
    if frame_indices is None:
        return None

    if torch.is_tensor(frame_indices):
        idx = frame_indices.to(device=device, dtype=torch.long)
    elif isinstance(frame_indices, (list, tuple)):
        idx = torch.tensor(frame_indices, device=device, dtype=torch.long)
    else:
        idx = torch.tensor([int(frame_indices)], device=device, dtype=torch.long)

    if idx.dim() == 0:
        idx = idx.view(1)

    return idx


def _motion_weight(frames_gt, config, frames_all=None, frame_indices=None, prev_target_batch=None):
    """
    Peso por movimiento calculado desde GT.

    Prioridad:
    1. frames_all + frame_indices: usa frame t contra t-1 real.
    2. prev_target_batch: usa el target actual contra el target previo ya movido al device.
    3. fallback: usa diferencias internas del batch.

    Permite calcular movimiento real durante entrenamiento frame-by-frame.

    """
    lambda_motion = _get_float(config, "lambda_motion", 0.0)
    if lambda_motion <= 0.0:
        return None

    B = frames_gt.shape[0]
    device = frames_gt.device

    if frames_all is not None and frame_indices is not None:
        # Indexamos en el device real de frames_all para soportar CPU uint8,
        # GPU uint8 o GPU float32. Luego movemos al device del loss.
        idx_src = _normalizar_indices(frame_indices, frames_all.device)
        prev_idx = torch.clamp(idx_src - 1, min=0)

        curr = frames_all[idx_src].to(device=device, non_blocking=True)
        prev = frames_all[prev_idx].to(device=device, non_blocking=True)

        if curr.dtype == torch.uint8:
            curr = curr.float().div_(255.0)
        else:
            curr = curr.to(dtype=frames_gt.dtype)

        if prev.dtype == torch.uint8:
            prev = prev.float().div_(255.0)
        else:
            prev = prev.to(dtype=frames_gt.dtype)

        motion = torch.abs(curr - prev).mean(dim=-1, keepdim=True)

        # El frame 0 no tiene frame previo real.
        idx = idx_src.to(device=device)
        es_frame0 = (idx == 0).view(-1, 1, 1, 1)
        motion = torch.where(es_frame0, torch.zeros_like(motion), motion)

    elif prev_target_batch is not None:
        prev_target_batch = prev_target_batch.to(device=device, dtype=frames_gt.dtype)
        motion = torch.abs(frames_gt - prev_target_batch).mean(dim=-1, keepdim=True)

        if frame_indices is not None:
            idx = _normalizar_indices(frame_indices, device)
            es_frame0 = (idx == 0).view(-1, 1, 1, 1)
            motion = torch.where(es_frame0, torch.zeros_like(motion), motion)

    else:
        motion = torch.zeros(
            (B, frames_gt.shape[1], frames_gt.shape[2], 1),
            device=device,
            dtype=frames_gt.dtype,
        )
        if B > 1:
            motion[1:] = torch.abs(frames_gt[1:] - frames_gt[:-1]).mean(dim=-1, keepdim=True)

    umbral = _get_float(config, "motion_umbral", 0.0)
    if umbral > 0.0:
        motion = (motion - umbral).clamp_min(0.0)

    motion_blur = int(config.get("motion_blur", 0))
    if motion_blur > 0:
        k = motion_blur if motion_blur % 2 == 1 else motion_blur + 1
        m = motion.permute(0, 3, 1, 2)
        m = F.avg_pool2d(m, kernel_size=k, stride=1, padding=k // 2)
        motion = m.permute(0, 2, 3, 1)

    motion_mean = motion.mean().detach()
    if motion_mean <= _EPS:
        motion_norm = torch.zeros_like(motion)
    else:
        motion_norm = motion / motion_mean.clamp_min(_EPS)

    clip_val = _get_float(config, "motion_clip", 5.0)
    motion_norm = motion_norm.clamp(0.0, clip_val)

    return (1.0 + lambda_motion * motion_norm).detach()


def _hard_weight(diff_abs, config):
    lambda_hard = _get_float(config, "lambda_hard", 0.0)
    if lambda_hard <= 0.0:
        return None

    err = diff_abs.mean(dim=-1, keepdim=True)
    err_mean = err.mean().detach()

    if err_mean <= _EPS:
        err_norm = torch.zeros_like(err)
    else:
        err_norm = err / err_mean.clamp_min(_EPS)

    clip_val = _get_float(config, "hard_clip", 5.0)
    err_norm = err_norm.clamp(0.0, clip_val)

    return (1.0 + lambda_hard * err_norm).detach()


def _sobel_mag(x_bchw):
    """x_bchw: (B,C,H,W). Devuelve magnitud Sobel en gris: (B,1,H,W)."""
    gray = x_bchw.mean(dim=1, keepdim=True)
    device = gray.device
    dtype = gray.dtype

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + _EPS)


def _loss_edge(render_batch, frames_gt):
    r = render_batch.permute(0, 3, 1, 2).clamp(0, 1)
    g = frames_gt.permute(0, 3, 1, 2).clamp(0, 1)
    return torch.mean(torch.abs(_sobel_mag(r) - _sobel_mag(g)))


def _loss_temporal(render_batch, frames_gt, prev_render_batch=None, prev_target_batch=None):
    """
    Loss temporal:
    - Si trainer pasa prev_render_batch y prev_target_batch:
        compara render[t] - render[t-1] contra gt[t] - gt[t-1].
    - Si no, usa diferencias internas del batch.
    """
    if prev_render_batch is not None and prev_target_batch is not None:
        delta_render = render_batch - prev_render_batch
        delta_gt = frames_gt - prev_target_batch
        return torch.mean(torch.abs(delta_render - delta_gt))

    if render_batch.shape[0] < 2:
        return render_batch.new_tensor(0.0)

    delta_render = render_batch[1:] - render_batch[:-1]
    delta_gt = frames_gt[1:] - frames_gt[:-1]
    return torch.mean(torch.abs(delta_render - delta_gt))



def loss_render_frame(
    render_hw3,
    target_hw3,
    config_or_lambda=0.2,
    lambda_dssim=None,
    frames_all=None,
    frame_idx=None,
    prev_render=None,
    prev_target=None,
):
    """
    Loss para un frame.

    Uso recomendado:
        loss_render_frame(render_j, frames[j], config, frames_all=frames, frame_idx=j)

    Compatible con el codigo anterior:
        loss_render_frame(render_j, frames[j], lambda_dssim=0.2)
    """
    rb = render_hw3.unsqueeze(0)
    tb = target_hw3.unsqueeze(0)

    frame_indices = None if frame_idx is None else [int(frame_idx)]

    prev_rb = None if prev_render is None else prev_render.unsqueeze(0)
    prev_tb = None if prev_target is None else prev_target.unsqueeze(0)

    return loss_render_batch(
        rb,
        tb,
        config_or_lambda=config_or_lambda,
        lambda_dssim=lambda_dssim,
        frames_all=frames_all,
        frame_indices=frame_indices,
        prev_render_batch=prev_rb,
        prev_target_batch=prev_tb,
    )


def loss_render_batch(
    render_batch,
    frames_gt,
    config_or_lambda=0.2,
    lambda_dssim=None,
    frames_all=None,
    frame_indices=None,
    prev_render_batch=None,
    prev_target_batch=None,
):
    """
    Loss principal para batch o frame.

    Args:
        render_batch: (B,H,W,3)
        frames_gt: (B,H,W,3)
        config_or_lambda:
            - dict config completo
            - float lambda_dssim para compatibilidad vieja
        frames_all + frame_indices:
            necesarios para motion real t contra t-1.
        prev_render_batch + prev_target_batch:
            necesarios para temporal real cuando se entrena frame por frame.
    """
    config = _config_from_arg(config_or_lambda, lambda_dssim=lambda_dssim)
    tipo = str(config.get("tipo_loss", "baseline")).lower().strip()

    diff_abs = torch.abs(render_batch - frames_gt)

    if tipo in ("baseline", "default", "l1", "l1_dssim"):
        loss = _aggregate_pixel(diff_abs, None, config)
        return _mezclar_dssim(loss, render_batch, frames_gt, config)

    if tipo == "l1_mse":
        loss = _aggregate_pixel(diff_abs, None, config)

        lambda_mse = _get_float(config, "lambda_mse", 0.0)
        if lambda_mse > 0.0:
            loss = loss + lambda_mse * torch.mean((render_batch - frames_gt) ** 2)

        return _mezclar_dssim(loss, render_batch, frames_gt, config)

    if tipo == "motion":
        w_motion = _motion_weight(
            frames_gt,
            config,
            frames_all=frames_all,
            frame_indices=frame_indices,
            prev_target_batch=prev_target_batch,
        )
        loss = _aggregate_pixel(diff_abs, w_motion, config)
        return _mezclar_dssim(loss, render_batch, frames_gt, config)

    if tipo == "hard":
        w_hard = _hard_weight(diff_abs, config)
        loss = _aggregate_pixel(diff_abs, w_hard, config)
        return _mezclar_dssim(loss, render_batch, frames_gt, config)

    if tipo == "edge":
        loss = _aggregate_pixel(diff_abs, None, config)

        lambda_edge = _get_float(config, "lambda_edge", 0.0)
        if lambda_edge > 0.0:
            loss = loss + lambda_edge * _loss_edge(render_batch, frames_gt)

        return _mezclar_dssim(loss, render_batch, frames_gt, config)

    if tipo == "temporal":
        loss = _aggregate_pixel(diff_abs, None, config)

        lambda_temporal = _get_float(config, "lambda_temporal", 0.0)
        if lambda_temporal > 0.0:
            loss = loss + lambda_temporal * _loss_temporal(
                render_batch,
                frames_gt,
                prev_render_batch=prev_render_batch,
                prev_target_batch=prev_target_batch,
            )

        return _mezclar_dssim(loss, render_batch, frames_gt, config)

    if tipo == "motion_temporal":
        w_motion = _motion_weight(
            frames_gt,
            config,
            frames_all=frames_all,
            frame_indices=frame_indices,
            prev_target_batch=prev_target_batch,
        )
        loss = _aggregate_pixel(diff_abs, w_motion, config)

        lambda_mse = _get_float(config, "lambda_mse", 0.0)
        if lambda_mse > 0.0:
            loss = loss + lambda_mse * torch.mean((render_batch - frames_gt) ** 2)

        lambda_temporal = _get_float(config, "lambda_temporal", 0.0)
        if lambda_temporal > 0.0:
            loss = loss + lambda_temporal * _loss_temporal(
                render_batch,
                frames_gt,
                prev_render_batch=prev_render_batch,
                prev_target_batch=prev_target_batch,
            )

        return _mezclar_dssim(loss, render_batch, frames_gt, config)

    if tipo == "combo":
        weight = None

        if _get_float(config, "lambda_motion", 0.0) > 0.0:
            weight = _motion_weight(
                frames_gt,
                config,
                frames_all=frames_all,
                frame_indices=frame_indices,
                prev_target_batch=prev_target_batch,
            )

        if _get_float(config, "lambda_hard", 0.0) > 0.0:
            w_hard = _hard_weight(diff_abs, config)
            weight = w_hard if weight is None else weight * w_hard

        loss = _aggregate_pixel(diff_abs, weight, config)

        lambda_mse = _get_float(config, "lambda_mse", 0.0)
        if lambda_mse > 0.0:
            loss = loss + lambda_mse * torch.mean((render_batch - frames_gt) ** 2)

        lambda_edge = _get_float(config, "lambda_edge", 0.0)
        if lambda_edge > 0.0:
            loss = loss + lambda_edge * _loss_edge(render_batch, frames_gt)

        lambda_temporal = _get_float(config, "lambda_temporal", 0.0)
        if lambda_temporal > 0.0:
            loss = loss + lambda_temporal * _loss_temporal(
                render_batch,
                frames_gt,
                prev_render_batch=prev_render_batch,
                prev_target_batch=prev_target_batch,
            )

        return _mezclar_dssim(loss, render_batch, frames_gt, config)

    raise ValueError(
        f"tipo_loss desconocido: {tipo}. "
        "Usa baseline, l1_mse, motion, hard, edge, temporal, motion_temporal o combo."
    )


def loss_smoothness(modelo, pesos_por_param=None):
    """Penaliza coeficientes de orden alto usando factores cacheados."""
    total = None
    pesos_por_param = pesos_por_param or {}

    for nombre, (a0, a_high, grado, _) in modelo.parametros_temporales().items():
        peso = pesos_por_param.get(nombre, 1.0)

        if peso == 0.0 or grado == 0:
            continue

        factor = modelo.smooth_factors[nombre]

        if factor.device != a_high.device or factor.dtype != a_high.dtype:
            factor = factor.to(device=a_high.device, dtype=a_high.dtype)

        while factor.dim() < a_high.dim():
            factor = factor.unsqueeze(0)

        contrib = (a_high ** 2 * factor).sum() * peso
        total = contrib if total is None else total + contrib

    if total is None:
        return torch.tensor(0.0, device=modelo.mu_a0.device)

    return total
