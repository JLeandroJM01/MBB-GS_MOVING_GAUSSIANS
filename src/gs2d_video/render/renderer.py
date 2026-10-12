import torch

from gs2d_video.render.cuda_tiled import (
    rasterizar_un_frame_cuda_tiled,
    loss_un_frame_cuda_tiled,
)

def render_frame(params_j, H, W, config):
    if not bool(config.get("usar_cuda_tiled", True)):
        raise RuntimeError("El renderer requiere usar_cuda_tiled=true")

    return rasterizar_un_frame_cuda_tiled(
        params_j,
        H,
        W,
        tile_size=int(config.get("cuda_tile_size", 16)),
        k_sigma=float(config.get("cuda_k_sigma", 3.5)),
    )


def loss_frame_cuda(params_j, target_j, H, W, config):
    if not bool(config.get("usar_cuda_tiled", True)):
        raise RuntimeError("El renderer requiere usar_cuda_tiled=true")

    loss_type = config.get("loss_cuda_tipo", "l1")

    return loss_un_frame_cuda_tiled(
        params_j,
        target_j,
        H,
        W,
        tile_size=int(config.get("cuda_tile_size", 16)),
        k_sigma=float(config.get("cuda_k_sigma", 3.5)),
        loss_type=loss_type,
    )

@torch.no_grad()
def render_clip(modelo, matrices_base, H, W, n_frames, config):
    renders = []

    for j in range(n_frames):
        params_j = modelo.evaluar_en_frame(j, matrices_base)
        r = render_frame(params_j, H, W, config).clamp(0, 1)
        renders.append(r)

    return torch.stack(renders, dim=0)
