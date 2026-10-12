"""
Adam con param_groups: por TIPO de parametro Y por ORDEN del coeficiente.

Separacion de a_0 y a_high:
------------------------------------
El modelo guarda a_0 y a_high en nn.Parameters distintos por construccion
(ver modelo.py). Asi, construir los param_groups es simplemente listar los
12 parametros (6 tipos x {a0, high}) con sus lrs correspondientes.

Convencion de claves en config["lrs"]:
    <param>_a0    -> lr para el coeficiente constante
    <param>_high  -> lr para los coeficientes a_1..a_q (todos juntos)
"""
import torch



# Valores usados cuando una tasa de aprendizaje no esta definida
_DEFAULTS = {
    'mu_a0':      1e-3, 'mu_high':      1e-4,
    'opacity_a0': 5e-2, 'opacity_high': 5e-3,
    'color_a0':   1e-2, 'color_high':   1e-3,
    'scale_a0':   5e-3, 'scale_high':   5e-4,
    'theta_a0':   5e-3, 'theta_high':   5e-4,
    'depth_a0':   5e-3, 'depth_high':   5e-4,
}



def construir_optimizador(modelo, lrs):
    """
    Args:
        modelo : GaussianasPolinomial2D
        lrs    : dict con claves <param>_a0 / <param>_high. Si falta alguna,
                 usa el default.
    Returns:
        torch.optim.Adam con 12 param_groups nombrados.
    """
    def lr(clave):
        return lrs.get(clave, _DEFAULTS[clave])

    grupos = []
    for nombre in ['mu', 'opacity', 'color', 'scale', 'theta', 'depth']:
        a0 = getattr(modelo, f"{nombre}_a0")
        hi = getattr(modelo, f"{nombre}_high")
        grupos.append({'params': [a0], 'lr': lr(f"{nombre}_a0"),   'name': f"{nombre}_a0"})
        grupos.append({'params': [hi], 'lr': lr(f"{nombre}_high"), 'name': f"{nombre}_high"})

    try:
        return torch.optim.Adam(grupos, fused=True)
    except Exception as e:
        print(f"[optimizador] Adam fused no disponible ({type(e).__name__}); probando foreach=True")

    try:
        return torch.optim.Adam(grupos, foreach=True)
    except Exception as e:
        print(f"[optimizador] Adam foreach no disponible ({type(e).__name__}); usando Adam normal")

    return torch.optim.Adam(grupos)
