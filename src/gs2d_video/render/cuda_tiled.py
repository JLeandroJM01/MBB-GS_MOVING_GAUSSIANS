import os
import sys
import math
import torch

_YA_IMPRIMIO_PREPROCESS = False
_YA_IMPRIMIO_CONIC = False
_YA_IMPRIMIO_GRAD_CONIC = False

AQUI = os.path.dirname(os.path.abspath(__file__))

RAIZ_REPO = os.path.abspath(
    os.path.join(AQUI, "..", "..", "..")
)

RUTA_RASTER_CUDA = os.environ.get(
    "RUTA_RASTER_CUDA",
    os.path.join(RAIZ_REPO, "cuda", "raster_cuda")
)

print(f"[rasterizador_cuda_tiled_autograd] buscando raster_cuda en: {RUTA_RASTER_CUDA}", flush=True)

if not os.path.isdir(RUTA_RASTER_CUDA):
    raise FileNotFoundError(f"No existe la carpeta raster_cuda: {RUTA_RASTER_CUDA}")

# Append para no tapar los wrappers del experimento.
if RUTA_RASTER_CUDA not in sys.path:
    sys.path.append(RUTA_RASTER_CUDA)

try:
    import raster_cuda
except Exception as e:
    print("[rasterizador_cuda_tiled_autograd] sys.path usado:", flush=True)
    for p in sys.path[:8]:
        print("  ", p, flush=True)
    raise e


def construir_conic(scale, theta):
    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    inv_sx2 = 1.0 / (sx * sx + 1e-8)
    inv_sy2 = 1.0 / (sy * sy + 1e-8)

    m00 = c * c * inv_sx2 + s * s * inv_sy2
    m01 = c * s * (inv_sx2 - inv_sy2)
    m11 = s * s * inv_sx2 + c * c * inv_sy2

    return torch.stack([m00, m01, m11], dim=1).contiguous()


def grad_conic_a_scale_theta(scale, theta, grad_conic):
    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    den_x = sx * sx + 1e-8
    den_y = sy * sy + 1e-8

    A = 1.0 / den_x
    B = 1.0 / den_y

    gm00 = grad_conic[:, 0]
    gm01 = grad_conic[:, 1]
    gm11 = grad_conic[:, 2]

    grad_A = gm00 * c * c + gm01 * c * s + gm11 * s * s
    grad_B = gm00 * s * s - gm01 * c * s + gm11 * c * c

    dA_dsx = -2.0 * sx / (den_x * den_x)
    dB_dsy = -2.0 * sy / (den_y * den_y)

    grad_sx = grad_A * dA_dsx
    grad_sy = grad_B * dB_dsy

    grad_scale = torch.stack([grad_sx, grad_sy], dim=1)

    dm00_dtheta = 2.0 * c * s * (B - A)
    dm01_dtheta = (c * c - s * s) * (A - B)
    dm11_dtheta = 2.0 * c * s * (A - B)

    grad_theta = (
        gm00 * dm00_dtheta +
        gm01 * dm01_dtheta +
        gm11 * dm11_dtheta
    )

    return grad_scale, grad_theta


def calcular_tiles_tocados_gpu(mu, scale, theta, depth, H, W, tile_size=16, k_sigma=3.5):
    device = mu.device
    N = mu.shape[0]

    tiles_x = math.ceil(W / tile_size)
    tiles_y = math.ceil(H / tile_size)
    total_tiles = tiles_x * tiles_y

    fila = mu[:, 0]
    col = mu[:, 1]

    sx = scale[:, 0]
    sy = scale[:, 1]

    c = torch.cos(theta)
    s = torch.sin(theta)

    extent_fila = k_sigma * torch.sqrt((c * sx) ** 2 + (s * sy) ** 2)
    extent_col = k_sigma * torch.sqrt((s * sx) ** 2 + (c * sy) ** 2)

    min_f = torch.floor(fila - extent_fila).clamp(0, H - 1).to(torch.int64)
    max_f = torch.ceil(fila + extent_fila).clamp(0, H - 1).to(torch.int64)
    min_c = torch.floor(col - extent_col).clamp(0, W - 1).to(torch.int64)
    max_c = torch.ceil(col + extent_col).clamp(0, W - 1).to(torch.int64)

    tile_y_min = torch.div(min_f, tile_size, rounding_mode="floor")
    tile_y_max = torch.div(max_f, tile_size, rounding_mode="floor")
    tile_x_min = torch.div(min_c, tile_size, rounding_mode="floor")
    tile_x_max = torch.div(max_c, tile_size, rounding_mode="floor")

    num_y = tile_y_max - tile_y_min + 1
    num_x = tile_x_max - tile_x_min + 1
    counts = num_y * num_x

    total_instancias = int(counts.sum().item())

    gaussian_ids = torch.repeat_interleave(
        torch.arange(N, device=device, dtype=torch.int64),
        counts
    )

    starts = torch.cumsum(counts, dim=0) - counts
    local_offsets = (
        torch.arange(total_instancias, device=device, dtype=torch.int64)
        - starts[gaussian_ids]
    )

    nx = num_x[gaussian_ids]
    local_y = torch.div(local_offsets, nx, rounding_mode="floor")
    local_x = local_offsets - local_y * nx

    tx = tile_x_min[gaussian_ids] + local_x
    ty = tile_y_min[gaussian_ids] + local_y

    tile_ids = ty * tiles_x + tx

    depth_vals = depth[gaussian_ids]
    d_min = depth_vals.min()
    d_max = depth_vals.max()
    depth_norm = (depth_vals - d_min) / (d_max - d_min + 1e-8)

    DEPTH_SCALE = 1_000_000
    depth_q = torch.clamp(
        (depth_norm * (DEPTH_SCALE - 1)).to(torch.int64),
        0,
        DEPTH_SCALE - 1
    )

    keys = tile_ids * DEPTH_SCALE + depth_q
    order = torch.argsort(keys)

    tile_ids = tile_ids[order].contiguous()
    gaussian_ids = gaussian_ids[order].contiguous()

    ranges = torch.full((total_tiles, 2), -1, device=device, dtype=torch.int64)

    if tile_ids.numel() > 0:
        is_start = torch.empty_like(tile_ids, dtype=torch.bool)
        is_start[0] = True
        is_start[1:] = tile_ids[1:] != tile_ids[:-1]

        starts_idx = torch.nonzero(is_start, as_tuple=False).flatten()
        unique_tiles = tile_ids[starts_idx]

        ends_idx = torch.empty_like(starts_idx)
        if starts_idx.numel() > 1:
            ends_idx[:-1] = starts_idx[1:]
        ends_idx[-1] = tile_ids.numel()

        ranges[unique_tiles, 0] = starts_idx
        ranges[unique_tiles, 1] = ends_idx

    return gaussian_ids.contiguous(), ranges.contiguous()


class RasterizarTiledCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, scale, theta, opacity, color, depth, H, W, tile_size, k_sigma):
        mu_c = mu.contiguous()
        scale_c = scale.contiguous()
        theta_c = theta.contiguous()
        opacity_c = opacity.contiguous()
        color_c = color.contiguous()
        depth_c = depth.contiguous()

        global _YA_IMPRIMIO_CONIC
        if hasattr(raster_cuda, "build_conic"):
            if not _YA_IMPRIMIO_CONIC:
                print("[tiled] usando build_conic CUDA/C++", flush=True)
                _YA_IMPRIMIO_CONIC = True
            conic = raster_cuda.build_conic(scale_c, theta_c)
        else:
            if not _YA_IMPRIMIO_CONIC:
                print("[tiled] usando build_conic PyTorch fallback", flush=True)
                _YA_IMPRIMIO_CONIC = True
            conic = construir_conic(scale_c, theta_c)

        usar_preprocess_cuda = os.environ.get("USAR_PREPROCESS_TILED_CUDA", "1") == "1"

        global _YA_IMPRIMIO_PREPROCESS
        if usar_preprocess_cuda and hasattr(raster_cuda, "preprocess_tiled"):
            if not _YA_IMPRIMIO_PREPROCESS:
                print("[tiled] usando preprocess CUDA/C++", flush=True)
                _YA_IMPRIMIO_PREPROCESS = True

            gaussian_ids, ranges = raster_cuda.preprocess_tiled(
                mu_c,
                scale_c,
                theta_c,
                depth_c,
                H,
                W,
                tile_size,
                k_sigma
            )
        else:
            if not _YA_IMPRIMIO_PREPROCESS:
                print("[tiled] usando preprocess PyTorch fallback", flush=True)
                _YA_IMPRIMIO_PREPROCESS = True

            gaussian_ids, ranges = calcular_tiles_tocados_gpu(
                mu_c,
                scale_c,
                theta_c,
                depth_c,
                H,
                W,
                tile_size=tile_size,
                k_sigma=k_sigma
            )

        render, final_Ts, n_contrib = raster_cuda.forward_tiled_train(
            mu_c,
            conic,
            opacity_c,
            color_c,
            gaussian_ids,
            ranges,
            H,
            W,
            tile_size
        )

        ctx.save_for_backward(
            mu_c,
            scale_c,
            theta_c,
            conic,
            opacity_c,
            color_c,
            gaussian_ids,
            ranges,
            final_Ts,
            n_contrib
        )
        ctx.H = H
        ctx.W = W
        ctx.tile_size = tile_size

        return render

    @staticmethod
    def backward(ctx, grad_render):
        (
            mu,
            scale,
            theta,
            conic,
            opacity,
            color,
            gaussian_ids,
            ranges,
            final_Ts,
            n_contrib
        ) = ctx.saved_tensors

        grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_tiled_fast(
            mu,
            conic,
            opacity,
            color,
            gaussian_ids,
            ranges,
            final_Ts,
            n_contrib,
            grad_render.contiguous(),
            ctx.H,
            ctx.W,
            ctx.tile_size
        )

        global _YA_IMPRIMIO_GRAD_CONIC
        if hasattr(raster_cuda, "grad_conic_to_scale_theta"):
            if not _YA_IMPRIMIO_GRAD_CONIC:
                print("[tiled] usando grad_conic_to_scale_theta CUDA/C++", flush=True)
                _YA_IMPRIMIO_GRAD_CONIC = True
            grad_scale, grad_theta = raster_cuda.grad_conic_to_scale_theta(
                scale,
                theta,
                grad_conic
            )
        else:
            if not _YA_IMPRIMIO_GRAD_CONIC:
                print("[tiled] usando grad_conic_to_scale_theta PyTorch fallback", flush=True)
                _YA_IMPRIMIO_GRAD_CONIC = True
            grad_scale, grad_theta = grad_conic_a_scale_theta(
                scale,
                theta,
                grad_conic
            )

        return (
            grad_mu,
            grad_scale,
            grad_theta,
            grad_opacity,
            grad_color,
            None,
            None,
            None,
            None,
            None
        )


class RasterizarTiledLossCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, scale, theta, opacity, color, depth, target, H, W, tile_size, k_sigma, loss_type):
        mu_c = mu.contiguous()
        scale_c = scale.contiguous()
        theta_c = theta.contiguous()
        opacity_c = opacity.contiguous()
        color_c = color.contiguous()
        depth_c = depth.contiguous()
        target_c = target.contiguous()

        conic = raster_cuda.build_conic(scale_c, theta_c)

        gaussian_ids, ranges = raster_cuda.preprocess_tiled(
            mu_c,
            scale_c,
            theta_c,
            depth_c,
            H,
            W,
            tile_size,
            k_sigma
        )

        loss, grad_render, final_Ts, n_contrib = raster_cuda.forward_tiled_train_loss(
            mu_c,
            conic,
            opacity_c,
            color_c,
            gaussian_ids,
            ranges,
            target_c,
            H,
            W,
            tile_size,
            loss_type
        )

        ctx.save_for_backward(
            mu_c,
            scale_c,
            theta_c,
            conic,
            opacity_c,
            color_c,
            gaussian_ids,
            ranges,
            final_Ts,
            n_contrib,
            grad_render
        )

        ctx.H = H
        ctx.W = W
        ctx.tile_size = tile_size

        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        (
            mu,
            scale,
            theta,
            conic,
            opacity,
            color,
            gaussian_ids,
            ranges,
            final_Ts,
            n_contrib,
            grad_render
        ) = ctx.saved_tensors

        grad_render_scaled = grad_render
        if grad_loss is not None:
            grad_render_scaled = grad_render * grad_loss.to(device=grad_render.device, dtype=grad_render.dtype)

        grad_mu, grad_conic, grad_opacity, grad_color = raster_cuda.backward_tiled_fast(
            mu,
            conic,
            opacity,
            color,
            gaussian_ids,
            ranges,
            final_Ts,
            n_contrib,
            grad_render_scaled.contiguous(),
            ctx.H,
            ctx.W,
            ctx.tile_size
        )

        grad_scale, grad_theta = raster_cuda.grad_conic_to_scale_theta(
            scale,
            theta,
            grad_conic
        )

        return (
            grad_mu,
            grad_scale,
            grad_theta,
            grad_opacity,
            grad_color,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

def rasterizar_un_frame_cuda_tiled(params_frame, H, W, tile_size=16, k_sigma=3.5):
    # El preprocess CUDA ya ordena por tile + depth.
    mu = params_frame["mu"].contiguous()
    scale = params_frame["scale"].contiguous()
    theta = params_frame["theta"].contiguous()
    opacity = params_frame["opacity"].contiguous()
    color = params_frame["color"].contiguous()
    depth = params_frame["depth"].contiguous()

    return RasterizarTiledCUDA.apply(
        mu,
        scale,
        theta,
        opacity,
        color,
        depth,
        H,
        W,
        tile_size,
        k_sigma
    )


def loss_un_frame_cuda_tiled(
    params_frame,
    target_frame,
    H,
    W,
    tile_size=16,
    k_sigma=3.5,
    loss_type="l1",
):
    mu = params_frame["mu"].contiguous()
    scale = params_frame["scale"].contiguous()
    theta = params_frame["theta"].contiguous()
    opacity = params_frame["opacity"].contiguous()
    color = params_frame["color"].contiguous()
    depth = params_frame["depth"].contiguous()
    target = target_frame.contiguous()

    return RasterizarTiledLossCUDA.apply(
        mu,
        scale,
        theta,
        opacity,
        color,
        depth,
        target,
        H,
        W,
        tile_size,
        k_sigma,
        loss_type,
    )