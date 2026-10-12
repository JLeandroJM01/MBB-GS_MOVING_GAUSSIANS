"""
Cuantizacion post-entrenamiento de los parametros Gabor.

mu_t se almacena como entero porque representa posiciones en samples y puede
alcanzar valores donde float16 pierde demasiada precision. Los parametros de
menor rango pueden evaluarse en float16.

Esquemas:
    fp32     : todos los parametros en float32
    fp16mix  : mu_t int32 y el resto float16
    fp16full : todos los parametros en float16
"""
import torch

_PARAMS = ["mu_t", "log_sigma", "amp", "freq_raw", "phi"]

# bytes por parametro segun tipo de almacenamiento
_BYTES = {"fp32": 4, "fp16": 2, "int": 4}

ESQUEMAS = {
    "fp32":    {p: "fp32" for p in _PARAMS},
    "fp16mix": {"mu_t": "int", "log_sigma": "fp16", "amp": "fp16",
                "freq_raw": "fp16", "phi": "fp16"},
    "fp16full": {p: "fp16" for p in _PARAMS},
}


def bytes_por_atomo(esquema):
    return sum(_BYTES[esquema[p]] for p in _PARAMS)


@torch.no_grad()
def render_cuantizado(modelo, esquema):
    """
    Aplica la cuantizacion del `esquema` a los parametros RAW del modelo,
    renderiza la waveform con esos valores degradados y RESTAURA los originales.
    Devuelve x_hat [T] (detached). No modifica el modelo de forma permanente.
    """
    originales = {p: getattr(modelo, p).detach().clone() for p in _PARAMS}
    try:
        for p in _PARAMS:
            tensor = getattr(modelo, p)
            modo = esquema[p]
            if modo == "fp16":
                tensor.data = tensor.data.half().float()
            elif modo == "int":
                tensor.data = tensor.data.round()
            # "fp32": sin cambios
        x_hat = modelo.render().detach()
    finally:
        for p in _PARAMS:
            getattr(modelo, p).data = originales[p]
    return x_hat
