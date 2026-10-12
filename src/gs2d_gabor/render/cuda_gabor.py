"""
Wrapper autograd para el Gabor splatting de audio.

Provee:
  - render_gabor_cuda(mu, sigma, amp, fnorm, phi, T, k_sigma) -> x_hat [T]
    usando la extension CUDA si esta disponible.
  - render_gabor_pytorch(...) -> fallback puro PyTorch (lento, denso) para
    validacion en CPU o cuando no hay CUDA. NO usarlo para entrenamiento real
    de audio largo: materializa una matriz [N, T].

Convencion de parametros (ya activados, tal como los espera el kernel):
    mu    : [N]  posicion temporal en samples
    sigma : [N]  ancho gaussiano en samples (> 0)
    amp   : [N]  amplitud (con signo)
    fnorm : [N]  frecuencia normalizada en ciclos/sample (0, 0.5)
    phi   : [N]  fase
    T     : int  numero de samples de la senal
    k_sigma : float  soporte local en multiplos de sigma
"""
import os
import sys
import math

import torch


_GABOR_CUDA = None
_GABOR_CUDA_ERROR = None
_YA_IMPRIMIO = False


def _intentar_cargar_cuda():
    global _GABOR_CUDA, _GABOR_CUDA_ERROR

    if _GABOR_CUDA is not None or _GABOR_CUDA_ERROR is not None:
        return

    aqui = os.path.dirname(os.path.abspath(__file__))
    raiz_repo = os.path.abspath(os.path.join(aqui, "..", "..", ".."))
    ruta = os.environ.get(
        "RUTA_GABOR_AUDIO_CUDA",
        os.path.join(raiz_repo, "cuda", "gabor_audio_cuda"),
    )

    if os.path.isdir(ruta) and ruta not in sys.path:
        sys.path.append(ruta)

    try:
        import gabor_audio_cuda  # noqa: F401
        _GABOR_CUDA = gabor_audio_cuda
    except Exception as e:  # pragma: no cover - depende del entorno CUDA
        _GABOR_CUDA_ERROR = e


class _RenderGaborCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, sigma, amp, fnorm, phi, T, k_sigma):
        mu_c = mu.contiguous()
        sigma_c = sigma.contiguous()
        amp_c = amp.contiguous()
        fnorm_c = fnorm.contiguous()
        phi_c = phi.contiguous()

        x_hat = _GABOR_CUDA.forward(mu_c, sigma_c, amp_c, fnorm_c, phi_c, int(T), float(k_sigma))

        ctx.save_for_backward(mu_c, sigma_c, amp_c, fnorm_c, phi_c)
        ctx.k_sigma = float(k_sigma)
        return x_hat

    @staticmethod
    def backward(ctx, grad_out):
        mu_c, sigma_c, amp_c, fnorm_c, phi_c = ctx.saved_tensors

        grad_mu, grad_sigma, grad_amp, grad_fnorm, grad_phi = _GABOR_CUDA.backward(
            mu_c, sigma_c, amp_c, fnorm_c, phi_c, grad_out.contiguous(), ctx.k_sigma
        )

        # T y k_sigma no reciben gradiente.
        return grad_mu, grad_sigma, grad_amp, grad_fnorm, grad_phi, None, None


def render_gabor_cuda(mu, sigma, amp, fnorm, phi, T, k_sigma=4.0):
    """Render Gabor usando la extension CUDA. Lanza si no esta disponible."""
    global _YA_IMPRIMIO

    _intentar_cargar_cuda()
    if _GABOR_CUDA is None:
        raise RuntimeError(
            f"gabor_audio_cuda no disponible: {_GABOR_CUDA_ERROR}. "
            f"Compila con: cd cuda/gabor_audio_cuda && python setup.py build_ext --inplace"
        )

    if not _YA_IMPRIMIO:
        print("[gabor] usando render CUDA", flush=True)
        _YA_IMPRIMIO = True

    return _RenderGaborCUDA.apply(mu, sigma, amp, fnorm, phi, int(T), float(k_sigma))


def render_gabor_pytorch(mu, sigma, amp, fnorm, phi, T, k_sigma=4.0, chunk_t=None):
    """
    Fallback puro PyTorch, diferenciable. Denso: materializa [N, T_chunk].

    Solo para validacion / CPU / audio corto. Para audio largo usa CUDA.

    chunk_t: si se da, procesa el eje temporal en bloques de ese tamano para
             acotar la memoria. Si None, hace el render denso completo.
    """
    device = mu.device
    dtype = mu.dtype
    N = mu.shape[0]

    omega = (2.0 * math.pi) * fnorm  # [N]

    if chunk_t is None:
        t = torch.arange(T, device=device, dtype=dtype).view(1, T)        # [1, T]
        d = t - mu.view(N, 1)                                             # [N, T]
        env = torch.exp(-0.5 * (d / sigma.view(N, 1).clamp_min(1e-6)) ** 2)
        osc = torch.cos(omega.view(N, 1) * d + phi.view(N, 1))
        contrib = amp.view(N, 1) * env * osc                             # [N, T]
        return contrib.sum(dim=0)                                         # [T]

    out = torch.zeros(T, device=device, dtype=dtype)
    for inicio in range(0, T, int(chunk_t)):
        fin = min(inicio + int(chunk_t), T)
        t = torch.arange(inicio, fin, device=device, dtype=dtype).view(1, -1)
        d = t - mu.view(N, 1)
        env = torch.exp(-0.5 * (d / sigma.view(N, 1).clamp_min(1e-6)) ** 2)
        osc = torch.cos(omega.view(N, 1) * d + phi.view(N, 1))
        contrib = amp.view(N, 1) * env * osc
        out[inicio:fin] = contrib.sum(dim=0)
    return out


def cuda_disponible():
    _intentar_cargar_cuda()
    return _GABOR_CUDA is not None
