import pytest
import torch

from gs2d_video.io.checkpoints import cargar_modelo_desde_checkpoint
from gs2d_video.core.bases import (
    construir_matriz_chebyshev,
    construir_matriz_monomial,
)
from gs2d_video.core.modelo import GaussianasPolinomial2D


GRADOS = {
    "mu": 2,
    "opacity": 2,
    "color": 2,
    "scale": 2,
    "theta": 2,
    "depth": 2,
}


@pytest.mark.parametrize(
    ("base_temporal", "constructor"),
    [
        ("chebyshev", construir_matriz_chebyshev),
        ("monomial", construir_matriz_monomial),
    ],
)
def test_checkpoint_loader_respeta_base(
    tmp_path,
    base_temporal,
    constructor,
):
    modelo = GaussianasPolinomial2D(
        n_gaussianas=5,
        n_frames=6,
        grados=GRADOS,
        H=16,
        W=20,
        device=torch.device("cpu"),
        escala_inicial_px=2.0,
        frame_0_imagen=None,
        semilla=42,
    )

    checkpoint = (
        tmp_path
        / f"{base_temporal}.pt"
    )

    torch.save(
        {
            "state_dict_coefs": modelo.state_dict_coefs(),
            "config": {
                "base_temporal": base_temporal,
                "grados": GRADOS,
                "n_gaussianas_inicial": 5,
                "escala_inicial_px": 2.0,
                "seed": 42,
            },
        },
        checkpoint,
    )

    cargado, config, matrices, info = (
        cargar_modelo_desde_checkpoint(
            checkpoint,
            device="cpu",
        )
    )

    assert cargado.numero_gaussianas() == 5
    assert info["N"] == 5
    assert info["H"] == 16
    assert info["W"] == 20
    assert info["n_frames"] == 6
    assert info["base_temporal"] == base_temporal

    assert config["base_temporal"] == base_temporal

    referencia = constructor(
        6,
        2,
        device="cpu",
        dtype=torch.float32,
    )

    assert torch.allclose(
        matrices[2],
        referencia,
    )

    for nombre in [
        "mu",
        "opacity",
        "color",
        "scale",
        "theta",
        "depth",
    ]:
        assert torch.equal(
            getattr(cargado, f"{nombre}_a0"),
            getattr(modelo, f"{nombre}_a0"),
        )

        assert torch.equal(
            getattr(cargado, f"{nombre}_high"),
            getattr(modelo, f"{nombre}_high"),
        )
