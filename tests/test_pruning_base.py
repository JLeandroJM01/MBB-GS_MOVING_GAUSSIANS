import torch

from gs2d_video.core.bases import (
    construir_matriz_chebyshev,
    construir_matriz_monomial,
)
from gs2d_video.core.modelo import GaussianasPolinomial2D
from gs2d_video.core.pruning_post import calcular_contribucion_maxima


GRADOS = {
    "mu": 3,
    "opacity": 3,
    "color": 3,
    "scale": 3,
    "theta": 3,
    "depth": 3,
}


def _modelo_con_opacidad(coefs_high):
    """Modelo de una sola gaussiana con los coeficientes de opacidad dados."""
    modelo = GaussianasPolinomial2D(
        n_gaussianas=1,
        n_frames=8,
        grados=GRADOS,
        H=8,
        W=8,
        device=torch.device("cpu"),
        escala_inicial_px=2.0,
        frame_0_imagen=None,
        semilla=0,
    )
    with torch.no_grad():
        modelo.opacity_a0.fill_(0.0)
        modelo.opacity_high.copy_(torch.tensor([coefs_high], dtype=torch.float32))
    return modelo


def test_contribucion_usa_la_base_que_se_le_pasa():
    """
    La opacidad se evalua sobre la base temporal con la que se entreno.

    En t = +-1 las dos bases coinciden (T_k(+-1) = (+-1)^k), asi que hay que
    elegir coeficientes cuyo maximo caiga en el interior para que las curvas
    se separen. Con a_2 = -1:
        Chebyshev -> -T_2(t) = 1 - 2t^2, maximo 1 en t = 0
        monomial  -> -t^2,               maximo 0 en t = 0
    """
    modelo = _modelo_con_opacidad([0.0, -1.0, 0.0])

    cheb = calcular_contribucion_maxima(
        modelo,
        n_samples=65,
        construir_matriz_base=construir_matriz_chebyshev,
    )
    mono = calcular_contribucion_maxima(
        modelo,
        n_samples=65,
        construir_matriz_base=construir_matriz_monomial,
    )

    assert torch.allclose(cheb[0], torch.sigmoid(torch.tensor(1.0)), atol=1e-6)
    assert torch.allclose(mono[0], torch.sigmoid(torch.tensor(0.0)), atol=1e-6)
    assert not torch.allclose(cheb, mono), (
        "evaluar en Chebyshev y en monomial dio el mismo resultado: "
        "la base pedida no se esta usando"
    )


def test_contribucion_monomial_coincide_con_evaluacion_directa():
    """
    Con base monomial, la contribucion debe ser max_t sigmoid(sum_k a_k t^k)
    calculado directamente sobre esa misma base.
    """
    coefs = [0.5, -1.0, 2.0]
    modelo = _modelo_con_opacidad(coefs)
    n_samples = 32

    obtenido = calcular_contribucion_maxima(
        modelo,
        n_samples=n_samples,
        construir_matriz_base=construir_matriz_monomial,
    )

    B = construir_matriz_monomial(n_samples, GRADOS["opacity"])
    a = torch.tensor([0.0] + coefs, dtype=torch.float32)
    esperado = torch.sigmoid(B @ a).max()

    assert torch.allclose(obtenido[0], esperado, atol=1e-6)


def test_por_defecto_sigue_siendo_chebyshev():
    """Sin argumento explicito se conserva el comportamiento anterior."""
    modelo = _modelo_con_opacidad([0.0, 1.5, -0.5])

    por_defecto = calcular_contribucion_maxima(modelo, n_samples=32)
    explicito = calcular_contribucion_maxima(
        modelo,
        n_samples=32,
        construir_matriz_base=construir_matriz_chebyshev,
    )

    assert torch.allclose(por_defecto, explicito)
