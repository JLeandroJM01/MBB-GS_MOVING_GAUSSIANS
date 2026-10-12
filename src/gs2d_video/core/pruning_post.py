"""
Pruning post-training basado en contribucion maxima de opacity (lottery
ticket): eliminar las gaussianas que nunca pasan el umbral en ningun frame.
"""
import torch

from gs2d_video.core.bases import construir_matriz_chebyshev


@torch.no_grad()
def calcular_contribucion_maxima(modelo, n_samples=200, construir_matriz_base=None):
    """
    max_t sigma(opacity_i(t)) muestreado en n_samples t en [0, n_frames-1].

    Args:
        modelo : GaussianasPolinomial2D
        n_samples : densidad temporal del muestreo
        construir_matriz_base : constructor de la base temporal con la que se
            entreno el modelo. Debe ser el mismo que uso el entrenamiento: si
            se evalua la opacidad en una base distinta de la aprendida, los
            valores no corresponden al modelo y se podan gaussianas
            equivocadas. Por defecto Chebyshev, que es la base principal.

    Returns: tensor (N,) en CPU con las contribuciones.
    """
    if construir_matriz_base is None:
        construir_matriz_base = construir_matriz_chebyshev

    grado_op = modelo.grados['opacity']
    device   = modelo.opacity_a0.device
    dtype    = modelo.opacity_a0.dtype

    B = construir_matriz_base(n_samples, grado_op, device=device, dtype=dtype)
    coefs = torch.cat([modelo.opacity_a0, modelo.opacity_high], dim=-1)   # (N, 1, grado+1)

    # (N, 1, grado+1) @ (grado+1, n_samples) = (N, 1, n_samples)
    raw = coefs @ B.T
    raw = raw.squeeze(1)                                                     # (N, n_samples)
    op = torch.sigmoid(raw)
    return op.max(dim=-1).values.cpu()



@torch.no_grad()
def prunear_post(modelo, umbral=0.05, n_samples=200, construir_matriz_base=None):
    """
    Filtra in-place todas las nn.Parameters por mascara (contrib >= umbral).

    `construir_matriz_base` debe ser la base temporal con la que se entreno el
    modelo; ver calcular_contribucion_maxima.

    Devuelve (n_original, n_final, indices_eliminados).
    """
    from torch import nn

    contribs = calcular_contribucion_maxima(
        modelo,
        n_samples=n_samples,
        construir_matriz_base=construir_matriz_base,
    )
    mantener = contribs >= umbral

    n_original = int(mantener.shape[0])
    n_final = int(mantener.sum().item())
    indices_eliminados = (~mantener).nonzero(as_tuple=False).squeeze(-1).tolist()

    if n_final == n_original:
        return n_original, n_final, indices_eliminados

    mascara = mantener.to(modelo.mu_a0.device)
    for nombre in ['mu', 'opacity', 'color', 'scale', 'theta', 'depth']:
        for sufijo in ['_a0', '_high']:
            attr = nombre + sufijo
            t = getattr(modelo, attr).data
            nuevo = nn.Parameter(t[mascara].clone())
            setattr(modelo, attr, nuevo)

    modelo.N = n_final
    return n_original, n_final, indices_eliminados
