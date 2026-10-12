import torch

from gs2d_video.core.bases import (
    construir_matriz_chebyshev,
    construir_matriz_monomial,
)
from gs2d_video.core.modelo import GaussianasPolinomial2D


def test_bases_temporales():
    n_frames = 7
    grado = 4

    cheb = construir_matriz_chebyshev(
        n_frames,
        grado,
        device="cpu",
        dtype=torch.float64,
    )

    mono = construir_matriz_monomial(
        n_frames,
        grado,
        device="cpu",
        dtype=torch.float64,
    )

    tau = torch.linspace(
        -1.0,
        1.0,
        n_frames,
        dtype=torch.float64,
    )

    assert cheb.shape == (n_frames, grado + 1)
    assert mono.shape == (n_frames, grado + 1)

    assert torch.allclose(
        cheb[:, 0],
        torch.ones_like(tau),
    )

    assert torch.allclose(
        cheb[:, 1],
        tau,
    )

    assert torch.allclose(
        cheb[:, 2],
        2.0 * tau**2 - 1.0,
    )

    for k in range(grado + 1):
        assert torch.allclose(
            mono[:, k],
            tau**k,
        )


def test_modelo_evaluacion_y_metadata():
    grados = {
        "mu": 2,
        "opacity": 2,
        "color": 2,
        "scale": 2,
        "theta": 2,
        "depth": 2,
    }

    modelo = GaussianasPolinomial2D(
        n_gaussianas=5,
        n_frames=6,
        grados=grados,
        H=16,
        W=20,
        device=torch.device("cpu"),
        escala_inicial_px=2.0,
        frame_0_imagen=None,
        semilla=42,
    )

    matrices = {
        2: construir_matriz_chebyshev(
            6,
            2,
            device="cpu",
            dtype=torch.float32,
        )
    }

    params = modelo.evaluar_en_frame(
        3,
        matrices,
    )

    assert params["mu"].shape == (5, 2)
    assert params["opacity"].shape == (5,)
    assert params["color"].shape == (5, 3)
    assert params["scale"].shape == (5, 2)
    assert params["theta"].shape == (5,)
    assert params["depth"].shape == (5,)

    for tensor in params.values():
        assert torch.isfinite(tensor).all()

    assert modelo.numero_gaussianas() == 5
    assert modelo.numero_gausianas() == 5

    sd = modelo.state_dict_coefs()

    assert int(sd["N"]) == 5
    assert int(sd["H"]) == 16
    assert int(sd["W"]) == 20
    assert int(sd["n_frames"]) == 6
    assert sd["grados"] == grados
