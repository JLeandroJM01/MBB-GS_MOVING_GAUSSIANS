import math

import torch


def render_denso(
    mu,
    sigma,
    amp,
    fnorm,
    phi,
    n_samples,
):
    n_atomos = mu.shape[0]

    t = torch.arange(
        n_samples,
        dtype=mu.dtype,
    ).view(1, n_samples)

    d = t - mu.view(n_atomos, 1)

    env = torch.exp(
        -0.5
        * (d / sigma.view(n_atomos, 1)) ** 2
    )

    osc = torch.cos(
        2.0
        * math.pi
        * fnorm.view(n_atomos, 1)
        * d
        + phi.view(n_atomos, 1)
    )

    return (
        amp.view(n_atomos, 1)
        * env
        * osc
    ).sum(dim=0)


def grad_manual(
    mu,
    sigma,
    amp,
    fnorm,
    phi,
    grad_out,
    n_samples,
):
    n_atomos = mu.shape[0]

    g_mu = torch.zeros(
        n_atomos,
        dtype=mu.dtype,
    )
    g_sigma = torch.zeros_like(g_mu)
    g_amp = torch.zeros_like(g_mu)
    g_f = torch.zeros_like(g_mu)
    g_phi = torch.zeros_like(g_mu)

    two_pi = 2.0 * math.pi

    for i in range(n_atomos):
        mu_i = mu[i].item()
        sigma_i = max(
            sigma[i].item(),
            1e-6,
        )
        amp_i = amp[i].item()
        omega_i = (
            two_pi
            * fnorm[i].item()
        )
        phi_i = phi[i].item()

        inv_sig2 = 1.0 / (
            sigma_i * sigma_i
        )
        inv_2sig2 = 0.5 * inv_sig2
        inv_sig3 = (
            inv_sig2
            / sigma_i
        )

        for t in range(n_samples):
            go = grad_out[t].item()
            d = float(t) - mu_i

            env = math.exp(
                -d
                * d
                * inv_2sig2
            )

            arg = (
                omega_i * d
                + phi_i
            )

            c = math.cos(arg)
            s = math.sin(arg)

            a_env = amp_i * env

            g_amp[i] += (
                go
                * env
                * c
            )

            g_mu[i] += (
                go
                * a_env
                * (
                    d
                    * inv_sig2
                    * c
                    + omega_i
                    * s
                )
            )

            g_sigma[i] += (
                go
                * a_env
                * (
                    d
                    * d
                    * inv_sig3
                )
                * c
            )

            g_f[i] += (
                go
                * a_env
                * (-s)
                * (
                    two_pi
                    * d
                )
            )

            g_phi[i] += (
                go
                * a_env
                * (-s)
            )

    return (
        g_mu,
        g_sigma,
        g_amp,
        g_f,
        g_phi,
    )


def test_gradientes_gabor_formula_vs_autograd():
    torch.manual_seed(0)

    n_atomos = 6
    n_samples = 200

    mu = (
        torch.rand(n_atomos)
        * n_samples
    ).double()

    sigma = (
        torch.rand(n_atomos)
        * 20
        + 5
    ).double()

    amp = (
        torch.rand(n_atomos)
        - 0.5
    ).double()

    fnorm = (
        torch.rand(n_atomos)
        * 0.4
        + 0.01
    ).double()

    phi = (
        torch.rand(n_atomos)
        * 2.0
        * math.pi
    ).double()

    grad_out = torch.randn(
        n_samples
    ).double()

    mu_a = mu.clone().requires_grad_(True)
    sigma_a = sigma.clone().requires_grad_(True)
    amp_a = amp.clone().requires_grad_(True)
    fnorm_a = fnorm.clone().requires_grad_(True)
    phi_a = phi.clone().requires_grad_(True)

    render = render_denso(
        mu_a,
        sigma_a,
        amp_a,
        fnorm_a,
        phi_a,
        n_samples,
    )

    render.backward(grad_out)

    manual = grad_manual(
        mu,
        sigma,
        amp,
        fnorm,
        phi,
        grad_out,
        n_samples,
    )

    autograd = (
        mu_a.grad,
        sigma_a.grad,
        amp_a.grad,
        fnorm_a.grad,
        phi_a.grad,
    )

    for grad_auto, grad_ref in zip(
        autograd,
        manual,
    ):
        assert torch.allclose(
            grad_auto,
            grad_ref,
            rtol=1e-6,
            atol=1e-8,
        )
