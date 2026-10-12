import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cargar_script(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_extraer_resuelve_raiz_del_repositorio():
    modulo = cargar_script(
        "extraer",
        ROOT / "scripts" / "data" / "extraer.py",
    )

    raiz = modulo.obtener_raiz_proyecto()
    _, video, clips, _ = modulo.obtener_rutas("video.mp4")

    assert raiz == ROOT
    assert video == ROOT / "data" / "videos" / "video.mp4"
    assert clips.parent == ROOT / "data" / "clips"


def test_extraer_clips_defaults():
    modulo = cargar_script(
        "extraer_clips_720p",
        ROOT / "scripts" / "data" / "extraer_clips_720p.py",
    )

    assert modulo.RAIZ == ROOT
    assert modulo.RUTA_VIDEO_DEFAULT == ROOT / "data" / "videos" / "video.mp4"
    assert modulo.RUTA_CLIPS_DEFAULT == ROOT / "data" / "clips"
