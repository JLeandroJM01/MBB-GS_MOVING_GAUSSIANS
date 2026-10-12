"""
Valida que la formula de gradientes del kernel CUDA (gabor_audio_cuda_kernel.cu)
coincide con autograd de PyTorch. Corre en CPU, no necesita GPU.

Replica EXACTAMENTE la matematica del backward kernel y la compara contra
torch.autograd sobre el mismo render denso. Si los gradientes coinciden, el
kernel CUDA esta matematicamente correcto.
"""
import math
import sys

import torch


def render_denso(mu, sigma, amp, fnorm, phi, T):
    """Render Gabor denso diferenciable (identico a render_gabor_pytorch sin chunk)."""
    N = mu.shape[0]
    t = torch.arange(T, dtype=mu.dtype).view(1, T)
    d = t - mu.view(N, 1)
    env = torch.exp(-0.5 * (d / sigma.view(N, 1)) ** 2)
    osc = torch.cos((2.0 * math.pi) * fnorm.view(N, 1) * d + phi.view(N, 1))
    return (amp.view(N, 1) * env * osc).sum(dim=0)


def grad_manual(mu, sigma, amp, fnorm, phi, grad_out, T):
    """
    Replica la formula del backward kernel CUDA (sin soporte local: k_sigma=inf,
    para comparar exacto con el render denso de autograd).
    """
    N = mu.shape[0]
    g_mu = torch.zeros(N)
    g_sigma = torch.zeros(N)
    g_amp = torch.zeros(N)
    g_f = torch.zeros(N)
    g_phi = torch.zeros(N)

    TWO_PI = 2.0 * math.pi

    for i in range(N):
        mu_i = mu[i].item()
        sigma_i = max(sigma[i].item(), 1e-6)
        amp_i = amp[i].item()
        omega_i = TWO_PI * fnorm[i].item()
        phi_i = phi[i].item()

        inv_sig2 = 1.0 / (sigma_i * sigma_i)
        inv_2sig2 = 0.5 * inv_sig2
        inv_sig3 = inv_sig2 / sigma_i

        for t in range(T):
            go = grad_out[t].item()
            d = float(t) - mu_i
            env = math.exp(-d * d * inv_2sig2)
            arg = omega_i * d + phi_i
            c = math.cos(arg)
            s = math.sin(arg)
            Aenv = amp_i * env

            g_amp[i] += go * env * c
            g_mu[i] += go * Aenv * (d * inv_sig2 * c + omega_i * s)
            g_sigma[i] += go * Aenv * (d * d * inv_sig3) * c
            g_f[i] += go * Aenv * (-s) * (TWO_PI * d)
            g_phi[i] += go * Aenv * (-s)

    return g_mu, g_sigma, g_amp, g_f, g_phi


def main():
    torch.manual_seed(0)
    N, T = 6, 200

    mu = (torch.rand(N) * T).double()
    sigma = (torch.rand(N) * 20 + 5).double()
    amp = (torch.rand(N) - 0.5).double()
    fnorm = (torch.rand(N) * 0.4 + 0.01).double()
    phi = (torch.rand(N) * 2 * math.pi).double()

    # gradiente de salida arbitrario (como si viniera del loss)
    grad_out = torch.randn(T).double()

    # --- autograd de referencia ---
    mu_a = mu.clone().requires_grad_(True)
    sigma_a = sigma.clone().requires_grad_(True)
    amp_a = amp.clone().requires_grad_(True)
    fnorm_a = fnorm.clone().requires_grad_(True)
    phi_a = phi.clone().requires_grad_(True)

    x_hat = render_denso(mu_a, sigma_a, amp_a, fnorm_a, phi_a, T)
    x_hat.backward(grad_out)

    # --- formula manual del kernel ---
    g_mu, g_sigma, g_amp, g_f, g_phi = grad_manual(mu, sigma, amp, fnorm, phi, grad_out, T)

    def comparar(nombre, auto, manual):
        err = (auto - manual).abs().max().item()
        rel = err / (auto.abs().max().item() + 1e-12)
        estado = "OK" if rel < 1e-6 else "FALLO"
        print(f"  {nombre:8s}  max_abs_err={err:.3e}  rel_err={rel:.3e}  [{estado}]")
        return rel < 1e-6

    print("Comparacion grad autograd vs formula del kernel CUDA:")
    ok = True
    ok &= comparar("mu",    mu_a.grad,    g_mu)
    ok &= comparar("sigma", sigma_a.grad, g_sigma)
    ok &= comparar("amp",   amp_a.grad,   g_amp)
    ok &= comparar("fnorm", fnorm_a.grad, g_f)
    ok &= comparar("phi",   phi_a.grad,   g_phi)

    print()
    if ok:
        print("TODOS OK: la matematica del backward kernel CUDA es correcta.")
    else:
        print("HAY DISCREPANCIAS: revisar la derivacion del kernel.")
        sys.exit(1)


if __name__ == "__main__":
    main()
