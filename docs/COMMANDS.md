# MBB-GS - Comandos principales

Este documento contiene los comandos principales del proyecto.

## Ir al proyecto

```powershell
cd MBB-GS_MOVING_GAUSSIANS
```

## Verificar PyTorch y CUDA

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Ejecutar tests

```powershell
python -m pytest -q
```

## Pipeline audiovisual

El pipeline procesa video, audio, entrenamiento, pruning, cuantizacion, metricas y reconstruccion.

```powershell
python scripts\pipeline\run_pipeline_video_audio.py --config configs\examples\smoke_20frames\pipeline.json
```

## Smoke test

El smoke test usa 20 frames y 1 epoch. Sirve para comprobar que el proyecto funciona, no para evaluar calidad.

```powershell
python scripts\pipeline\run_pipeline_video_audio.py --config configs\examples\smoke_20frames\pipeline.json
```

## Solo video

```text
python scripts\train.py --config RUTA_DE_LA_CONFIG
```

## Solo audio Gabor estereo

```text
python scripts\audio\train_gabor_stereo.py --config RUTA_DE_LA_CONFIG
```

## Nota

RUTA_DE_LA_CONFIG es un ejemplo y debe reemplazarse por una ruta real.

## Diagnostico del entorno

Comprueba Python, PyTorch, CUDA, GPU, NVCC, FFmpeg y las extensiones CUDA:

```powershell
python scripts\doctor.py
```

Para incluir tambien los tests:

```powershell
python scripts\doctor.py --tests
```

## Instalacion automatica en Windows

Desde PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

El instalador crea o reutiliza el entorno Conda, instala dependencias, detecta CUDA y Visual Studio, compila las extensiones y ejecuta el doctor.

## Generar medio sintetico para smoke

```powershell
python scripts\generate_smoke_media.py --force
```

## Instalacion automatica en Linux

```bash
chmod +x setup.sh
./setup.sh
```

## Smoke audiovisual reproducible

Generar el medio sintetico:

```powershell
python scripts\generate_smoke_media.py --force
```

Validar el entorno:

```powershell
python scripts\doctor.py --tests
```

Ejecutar el pipeline completo:

```powershell
python scripts\pipeline\run_pipeline_video_audio.py --config configs\examples\smoke_20frames\pipeline.json
```
