# MBB-GS - Troubleshooting

Este documento recopila problemas reales encontrados durante la instalacion y validacion de MBB-GS.

---

# 1. PyTorch no detecta CUDA

Comprobar:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Una instalacion funcional debe mostrar `True` en CUDA disponible.

Tambien puede comprobarse la GPU:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Si CUDA aparece como `False`, verificar que se instalo una version CUDA de PyTorch y no una version CPU.

---

# 2. Existen varias versiones de CUDA

Comprobar:

```powershell
where.exe nvcc
nvcc --version
```

En la maquina utilizada durante la validacion coexistian CUDA 12.6 y CUDA 11.8.

La version principal utilizada fue CUDA 12.6, compatible con PyTorch cu126.

Para evitar seleccionar una version incorrecta durante compilacion:

```cmd
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
```

---

# 3. cl.exe no se encuentra

Sintoma:

```text
where.exe cl
INFO: No se pudo encontrar ningun archivo
```

Esto puede ocurrir aunque Visual Studio tenga instalado el compilador C++.

Comprobar Visual Studio:

```powershell
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
```

En Windows se recomienda compilar desde:

```text
Visual Studio 2022 Developer Command Prompt
Environment initialized for: x64
```

Despues:

```cmd
where cl
```

Debe aparecer una ruta similar a:

```text
...\VC\Tools\MSVC\...\bin\Hostx64\x64\cl.exe
```

---

# 4. DISTUTILS_USE_SDK no esta configurado

Error observado:

```text
It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set.
```

Solucion en Developer Command Prompt:

```cmd
set DISTUTILS_USE_SDK=1
```

En PowerShell:

```powershell
$env:DISTUTILS_USE_SDK = "1"
```

Esta variable evita que PyTorch intente inicializar nuevamente el entorno de Visual Studio.

---

# 5. CUDAExtension compila pero el import falla

Sintoma:

```text
ImportError: DLL load failed while importing raster_cuda
```

El archivo `.pyd` puede haberse generado correctamente.

Comprobar:

```powershell
Get-ChildItem cuda\raster_cuda\*.pyd
```

En Windows, las extensiones dependen de DLL de PyTorch como:

```text
c10.dll
torch_cpu.dll
torch_python.dll
c10_cuda.dll
```

La solucion validada fue cargar PyTorch antes de importar la extension:

```powershell
python -c "import torch; import raster_cuda; print('raster_cuda OK')"
```

Para Gabor:

```powershell
python -c "import torch; import gabor_audio_cuda; print('gabor_audio_cuda OK')"
```

Los wrappers del proyecto ya importan PyTorch antes de cargar las extensiones.

---

# 6. Inspeccionar dependencias de un .pyd

Desde Visual Studio Developer Command Prompt:

```cmd
dumpbin /dependents raster_cuda.cp310-win_amd64.pyd
```

Esto permite identificar las DLL necesarias por la extension.

---

# 7. raster_cuda no existe

Comprobar:

```powershell
Get-ChildItem cuda\raster_cuda\*.pyd
```

Si no aparece ningun archivo, compilar:

```cmd
cd cuda\raster_cuda
set DISTUTILS_USE_SDK=1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
python setup.py build_ext --inplace
```

---

# 8. gabor_audio_cuda no existe

Comprobar:

```powershell
Get-ChildItem cuda\gabor_audio_cuda\*.pyd
```

Compilar:

```cmd
cd cuda\gabor_audio_cuda
set DISTUTILS_USE_SDK=1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
python setup.py build_ext --inplace
```

---

# 9. Warnings durante NVCC

Durante la compilacion pueden aparecer warnings de variables no utilizadas o warnings provenientes de headers de PyTorch.

Ejemplo observado:

```text
variable "s_gid" was set but never used
```

Si la compilacion termina creando el `.pyd` y no existe un error de compilacion, estos warnings no impiden utilizar la extension.

No se recomienda modificar kernels solamente para eliminar warnings.

---

# 10. FFmpeg no se encuentra

Comprobar:

```powershell
ffmpeg -version
ffprobe -version
```

Si alguno no existe, agregar la carpeta `bin` de FFmpeg al PATH.

El pipeline utiliza FFmpeg y FFprobe para inspeccion, extraccion y mux de audio/video.

---

# 11. El pipeline no encuentra el video

Ejemplo:

```text
FileNotFoundError
```

Revisar el campo:

```text
mp4_original
```

de `pipeline.json`.

Comprobar la ruta:

```powershell
Test-Path "data\videos\smoke_input.mp4"
```

Debe devolver:

```text
True
```

---

# 12. Config JSON invalida

Validar una configuracion:

```powershell
Get-Content "config.json" -Raw | ConvertFrom-Json | Out-Null
```

Validar los templates oficiales con Python:

```powershell
python -c "import json; from pathlib import Path; files=list(Path('configs/templates').glob('*.json')); [json.load(open(f, encoding='utf-8')) for f in files]; print('JSON OK', len(files))"
```

---

# 13. Se ejecuto literalmente EXPERIMENTO o RUTA_CONFIG

Textos como:

```text
EXP
RUTA_CONFIG
<nombre_experimento>
```

son marcadores de documentacion y deben reemplazarse por rutas reales.

Ejemplo real:

```powershell
python scripts\pipeline\run_pipeline_video_audio.py --config configs\examples\smoke_20frames\pipeline.json
```

---

# 14. PSNR igual a infinito

Si dos imagenes comparadas son exactamente iguales:

```text
MSE = 0
PSNR = inf
```

Esto puede ocurrir en pruning o cuantizacion cuando las diferencias numericas desaparecen despues de guardar los frames.

No significa que el modelo sea perfecto respecto al video original.

Significa que las dos reconstrucciones utilizadas por esa comparacion fueron identicas.

---

# 15. PSNR p5 o std aparecen como nan

En el estado actual, una lista compuesta solamente por valores infinitos puede producir warnings de NumPy y valores:

```text
p5 = nan
std = nan
```

Este comportamiento esta identificado para ser corregido durante la refactorizacion de metricas.

---

# 16. Pruning reutiliza resultados antiguos

Si se reutiliza exactamente el mismo nombre de experimento, pueden existir estadisticas, renders o resultados de pruning previos.

Hasta que el pipeline implemente limpieza y validacion de cache mas estrictas, se recomienda:

```text
usar nombres unicos para experimentos diferentes
o limpiar las salidas antes de una ejecucion reproducible
```

El smoke oficial utiliza `forzar_extraccion=true` para regenerar las entradas.

---

# 17. Tests

Ejecutar:

```powershell
python -m pytest -q
```

Si los tests fallan despues de modificar codigo, no continuar con un entrenamiento grande hasta revisar el error.

---

# 18. Comprobar sintaxis Python

Puede comprobarse todo el codigo Python con:

```powershell
python -m compileall -q src scripts tests
```

Sin salida significa que no se detectaron errores de compilacion de sintaxis.

---

# 19. Orden rapido de diagnostico

Ante una instalacion nueva o un error desconocido:

```text
1. python --version
2. import torch
3. torch.cuda.is_available()
4. nvcc --version
5. comprobar compilador C++
6. comprobar .pyd
7. comprobar FFmpeg
8. ejecutar pytest
9. ejecutar smoke test
```

---

# 20. Documentacion relacionada

```text
docs/INSTALLATION.md
docs/COMMANDS.md
docs/CONFIGURATION.md
docs/PIPELINE.md
```
