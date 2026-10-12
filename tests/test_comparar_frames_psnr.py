import importlib.util
import math
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "metrics"
    / "comparar_frames_psnr.py"
)

SPEC = importlib.util.spec_from_file_location("comparar_frames_psnr", SCRIPT)
MODULO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULO)


def test_psnr_identico_es_infinito():
    imagen = np.zeros((8, 8, 3), dtype=np.float32)

    psnr, mse, mae = MODULO.calcular_psnr(imagen, imagen)

    assert math.isinf(psnr)
    assert mse == 0.0
    assert mae == 0.0


def test_resumen_todos_inf():
    resumen = MODULO.resumir_psnr([float("inf")] * 20)

    assert math.isinf(resumen["promedio"])
    assert math.isinf(resumen["min"])
    assert math.isinf(resumen["p5"])
    assert math.isinf(resumen["max"])
    assert resumen["std"] == 0.0


def test_resumen_finito():
    resumen = MODULO.resumir_psnr([10.0, 20.0, 30.0, 40.0])

    assert resumen["promedio"] == 25.0
    assert resumen["min"] == 10.0
    assert resumen["max"] == 40.0
    assert math.isclose(resumen["p5"], 11.5)
    assert math.isclose(resumen["std"], np.std([10.0, 20.0, 30.0, 40.0]))
