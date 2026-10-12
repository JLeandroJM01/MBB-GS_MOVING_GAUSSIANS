from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ejecutar(comando):
    try:
        proceso = subprocess.run(
            comando,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        salida = (proceso.stdout or proceso.stderr).strip()
        return proceso.returncode, salida
    except Exception as exc:
        return 1, str(exc)


def primera_linea(texto):
    lineas = [linea.strip() for linea in texto.splitlines() if linea.strip()]
    return lineas[0] if lineas else ""


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostico rapido del entorno MBB-GS."
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help="Ejecuta pytest despues de revisar el entorno.",
    )
    args = parser.parse_args()

    resultados = []

    def registrar(nombre, estado, detalle=""):
        resultados.append((nombre, estado, detalle))

    version_python = platform.python_version()
    python_ok = sys.version_info >= (3, 10)
    registrar(
        "Python",
        "OK" if python_ok else "ERROR",
        f"{version_python} | {sys.executable}",
    )

    repo_ok = (ROOT / "pyproject.toml").is_file() and (ROOT / "src").is_dir()
    registrar("Repositorio", "OK" if repo_ok else "ERROR", str(ROOT))

    codigo_git, salida_git = ejecutar(
        ["git", "rev-parse", "--short", "HEAD"]
    )
    registrar(
        "Git commit",
        "OK" if codigo_git == 0 else "WARN",
        primera_linea(salida_git) if codigo_git == 0 else "no disponible",
    )

    try:
        import gs2d_video
        import gs2d_gabor

        registrar("Paquetes proyecto", "OK", "gs2d_video, gs2d_gabor")
    except Exception as exc:
        registrar("Paquetes proyecto", "ERROR", str(exc))

    torch_mod = None
    try:
        import torch

        torch_mod = torch
        registrar(
            "PyTorch",
            "OK",
            f"{torch.__version__} | build CUDA {torch.version.cuda}",
        )

        cuda_ok = torch.cuda.is_available()
        registrar(
            "CUDA PyTorch",
            "OK" if cuda_ok else "ERROR",
            str(cuda_ok),
        )

        if cuda_ok:
            registrar("GPU", "OK", torch.cuda.get_device_name(0))
        else:
            registrar("GPU", "ERROR", "GPU CUDA no disponible")
    except Exception as exc:
        registrar("PyTorch", "ERROR", str(exc))
        registrar("CUDA PyTorch", "ERROR", "PyTorch no disponible")
        registrar("GPU", "ERROR", "PyTorch no disponible")

    nvcc = shutil.which("nvcc")
    if nvcc:
        codigo, salida = ejecutar([nvcc, "--version"])
        detalle = primera_linea(salida)
        for linea in salida.splitlines():
            if "release" in linea.lower():
                detalle = linea.strip()
        registrar("NVCC", "OK" if codigo == 0 else "WARN", detalle)
    else:
        registrar("NVCC", "WARN", "no encontrado en PATH")

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        existe = Path(cuda_home).is_dir()
        registrar(
            "CUDA_HOME",
            "OK" if existe else "WARN",
            cuda_home,
        )
    else:
        registrar(
            "CUDA_HOME",
            "WARN",
            "no definido; necesario principalmente para compilacion",
        )

    if os.name == "nt":
        compilador = shutil.which("cl")
        if compilador:
            registrar("Compilador C++", "OK", compilador)
        else:
            registrar(
                "Compilador C++",
                "WARN",
                "cl.exe no visible; usar Developer Command Prompt x64 para compilar",
            )
    else:
        compilador = shutil.which("g++")
        if compilador:
            codigo, salida = ejecutar([compilador, "--version"])
            registrar(
                "Compilador C++",
                "OK" if codigo == 0 else "WARN",
                primera_linea(salida),
            )
        else:
            registrar("Compilador C++", "WARN", "g++ no encontrado")

    ninja = shutil.which("ninja")
    registrar(
        "Ninja",
        "OK" if ninja else "WARN",
        ninja or "no encontrado en PATH",
    )

    for programa in ("ffmpeg", "ffprobe"):
        ruta = shutil.which(programa)
        if ruta:
            codigo, salida = ejecutar([ruta, "-version"])
            registrar(
                programa.upper(),
                "OK" if codigo == 0 else "ERROR",
                primera_linea(salida),
            )
        else:
            registrar(programa.upper(), "ERROR", "no encontrado en PATH")

    raster_pyd = list((ROOT / "cuda" / "raster_cuda").glob("raster_cuda*.pyd"))
    raster_so = list((ROOT / "cuda" / "raster_cuda").glob("raster_cuda*.so"))
    raster_binario = raster_pyd + raster_so
    registrar(
        "Binario raster CUDA",
        "OK" if raster_binario else "ERROR",
        raster_binario[0].name if raster_binario else "no compilado",
    )

    try:
        from gs2d_video.render import cuda_tiled

        modulo = cuda_tiled.raster_cuda
        funciones = (
            "forward_tiled_train",
            "build_conic",
            "preprocess_tiled",
        )
        faltantes = [nombre for nombre in funciones if not hasattr(modulo, nombre)]
        if faltantes:
            registrar(
                "raster_cuda import",
                "ERROR",
                "faltan funciones: " + ", ".join(faltantes),
            )
        else:
            registrar("raster_cuda import", "OK", str(modulo.__file__))
    except Exception as exc:
        registrar("raster_cuda import", "ERROR", str(exc))

    gabor_pyd = list(
        (ROOT / "cuda" / "gabor_audio_cuda").glob("gabor_audio_cuda*.pyd")
    )
    gabor_so = list(
        (ROOT / "cuda" / "gabor_audio_cuda").glob("gabor_audio_cuda*.so")
    )
    gabor_binario = gabor_pyd + gabor_so
    registrar(
        "Binario Gabor CUDA",
        "OK" if gabor_binario else "ERROR",
        gabor_binario[0].name if gabor_binario else "no compilado",
    )

    try:
        from gs2d_gabor.render.cuda_gabor import cuda_disponible

        disponible = cuda_disponible()
        registrar(
            "gabor_audio_cuda import",
            "OK" if disponible else "ERROR",
            "extension disponible" if disponible else "extension no disponible",
        )
    except Exception as exc:
        registrar("gabor_audio_cuda import", "ERROR", str(exc))

    smoke_config = ROOT / "configs" / "examples" / "smoke_20frames" / "pipeline.json"
    if smoke_config.is_file():
        try:
            data = json.loads(smoke_config.read_text(encoding="utf-8-sig"))
            mp4 = ROOT / data["mp4_original"]
            estado = "OK" if mp4.is_file() else "WARN"
            detalle = str(mp4) if mp4.is_file() else f"video no encontrado: {mp4}"
            registrar("Smoke config", estado, detalle)
        except Exception as exc:
            registrar("Smoke config", "ERROR", str(exc))
    else:
        registrar("Smoke config", "WARN", "pipeline.json no encontrado")

    if args.tests:
        codigo, salida = ejecutar([sys.executable, "-m", "pytest", "-q"])
        detalle = primera_linea(salida)
        if salida:
            lineas_tests = [x.strip() for x in salida.splitlines() if x.strip()]
            detalle = lineas_tests[-1]
        registrar("Pytest", "OK" if codigo == 0 else "ERROR", detalle)

    print()
    print("=" * 78)
    print("MBB-GS ENVIRONMENT DOCTOR")
    print("=" * 78)

    ancho = max(len(nombre) for nombre, _, _ in resultados)
    for nombre, estado, detalle in resultados:
        print(f"{nombre:<{ancho}}  [{estado:<5}]  {detalle}")

    errores = sum(estado == "ERROR" for _, estado, _ in resultados)
    warnings = sum(estado == "WARN" for _, estado, _ in resultados)

    print("-" * 78)
    print(f"Errores: {errores} | Advertencias: {warnings}")

    if errores:
        print("Estado: entorno incompleto")
        return 1

    if warnings:
        print("Estado: funcional con advertencias")
        return 0

    print("Estado: entorno listo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
