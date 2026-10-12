# MBB-GS - Instalacion

Este documento describe la preparacion manual del entorno de MBB-GS.

El proyecto utiliza:

```text
Python 3.10
PyTorch con CUDA
CUDA Toolkit
compilador C++
FFmpeg
extensiones CUDA propias
```

Actualmente existen dos extensiones CUDA:

```text
cuda/raster_cuda       -> rasterizador de video
cuda/gabor_audio_cuda  -> renderer Gabor de audio
```

---

# 1. Requisitos generales

Se recomienda una GPU NVIDIA compatible con CUDA.

Software necesario:

```text
Git
Miniconda o Anaconda
Python 3.10
CUDA Toolkit
FFmpeg
compilador C++
```

En Windows se utiliza MSVC de Visual Studio.

En Linux se utiliza normalmente GCC/G++.

---

# 2. Instalacion en Windows

La configuracion validada durante la refactorizacion fue:

```text
Windows
Python 3.10
PyTorch 2.12.0+cu126
CUDA Toolkit 12.6
Visual Studio 2022 Community
MSVC x64
FFmpeg 8
```

La GPU utilizada para validar el proyecto fue una NVIDIA GeForce RTX 3050 6GB Laptop GPU.

---

# 3. Crear entorno Conda

Desde PowerShell o Anaconda Prompt:

```powershell
conda create -n mbb-gs python=3.10 -y
conda activate mbb-gs
```

Verificar:

```powershell
python --version
```

---

# 4. Instalar PyTorch CUDA

La instalacion validada utiliza CUDA 12.6.

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Verificar:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Una instalacion funcional debe mostrar:

```text
version de PyTorch
CUDA 12.6
True
```

---

# 5. Instalar dependencias del proyecto

Desde la raiz del repositorio:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

La instalacion editable permite importar directamente:

```text
gs2d_video
gs2d_gabor
```

Validar:

```powershell
python -c "import gs2d_video, gs2d_gabor; print('packages OK')"
```

---

# 6. CUDA Toolkit

Verificar que NVCC este disponible:

```powershell
nvcc --version
where.exe nvcc
```

Durante la validacion se utilizo:

```text
CUDA compilation tools, release 12.6
```

Si existen varias versiones de CUDA instaladas, debe asegurarse que la version utilizada para compilar sea compatible con PyTorch.

En Windows puede fijarse explicitamente:

```cmd
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
```

---

# 7. Visual Studio y MSVC

Las extensiones CUDA necesitan un compilador C++.

En Windows se utiliza Visual Studio 2022 con las herramientas C++ x64.

Una forma de comprobar la instalacion es:

```powershell
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
```

El proyecto fue validado con:

```text
Visual Studio 2022 Community
```

Para compilar manualmente se recomienda abrir:

```text
Visual Studio 2022 Developer Command Prompt
Environment initialized for: x64
```

Verificar:

```cmd
where cl
```

Debe aparecer una ruta hacia `cl.exe` dentro de Visual Studio.

---

# 8. Variables necesarias para compilar en Windows

Cuando el entorno de Visual Studio ya esta inicializado:

```cmd
set DISTUTILS_USE_SDK=1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
```

`DISTUTILS_USE_SDK=1` evita que PyTorch intente inicializar nuevamente el entorno MSVC.

---

# 9. Compilar raster_cuda

Desde Visual Studio Developer Command Prompt x64:

```cmd
cd cuda\raster_cuda
python setup.py build_ext --inplace
```

Debe generarse un archivo similar a:

```text
raster_cuda.cp310-win_amd64.pyd
```

Validar:

```cmd
python -c "import torch; import raster_cuda; print('raster_cuda OK')"
```

Es importante importar `torch` antes de la extension para que las DLL necesarias de PyTorch esten cargadas.

---

# 10. Compilar gabor_audio_cuda

```cmd
cd cuda\gabor_audio_cuda
python setup.py build_ext --inplace
```

Debe generarse un archivo similar a:

```text
gabor_audio_cuda.cp310-win_amd64.pyd
```

Validar:

```cmd
python -c "import torch; import gabor_audio_cuda; print('gabor_audio_cuda OK')"
```

---

# 11. FFmpeg

FFmpeg se utiliza para:

```text
inspeccionar MP4
extraer audio
codificar video
combinar video y audio
```

Verificar:

```powershell
ffmpeg -version
ffprobe -version
```

Ambos comandos deben estar disponibles desde PATH.

---

# 12. Ejecutar tests

Desde la raiz:

```powershell
python -m pytest -q
```

Durante la validacion de la refactorizacion:

```text
13 passed
```

---

# 13. Validacion CUDA

Ademas de compilar las extensiones, debe comprobarse que PyTorch detecta la GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Tambien es recomendable ejecutar el smoke audiovisual descrito mas adelante.

---

# 14. Smoke test audiovisual

La configuracion minima validada esta en:

```text
configs/examples/smoke_20frames/
```

El archivo de video indicado en `pipeline.json` debe existir antes de ejecutar el smoke.

Ejecutar:

```powershell
python scripts\pipeline\run_pipeline_video_audio.py --config configs\examples\smoke_20frames\pipeline.json
```

El smoke verifica:

```text
lectura del MP4
extraccion de frames
extraccion de audio
raster CUDA
Gabor CUDA
entrenamiento video
entrenamiento audio
pruning
UINT16
metricas
reconstruccion
FFmpeg
```

---

# 15. Instalacion en Linux

La estructura general es la misma que en Windows.

Se necesita:

```text
Python 3.10
PyTorch CUDA
CUDA Toolkit compatible
GCC/G++
FFmpeg
Ninja
```

Crear entorno:

```bash
conda create -n mbb-gs python=3.10 -y
conda activate mbb-gs
```

Instalar PyTorch y dependencias:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -e .
```

Configurar CUDA si fuera necesario:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Verificar:

```bash
nvcc --version
gcc --version
g++ --version
ffmpeg -version
```

---

# 16. Compilar CUDA en Linux

Raster:

```bash
cd cuda/raster_cuda
python setup.py build_ext --inplace
cd ../..
```

Gabor:

```bash
cd cuda/gabor_audio_cuda
python setup.py build_ext --inplace
cd ../..
```

Despues ejecutar:

```bash
python -m pytest -q
```

La compatibilidad exacta entre version de driver, PyTorch, CUDA Toolkit y compilador debe comprobarse en la maquina donde se instala.

---

# 17. Instalacion automatica

## Windows

El proyecto incluye `setup.ps1`, que automatiza:

```text
deteccion de Conda
creacion o reutilizacion del entorno mbb-gs
instalacion de PyTorch CUDA
instalacion de dependencias
deteccion de CUDA Toolkit
deteccion de Visual Studio C++
inicializacion automatica de MSVC x64
compilacion de raster_cuda
compilacion de gabor_audio_cuda
generacion del smoke sintetico
ejecucion de doctor.py
ejecucion de pytest
```

Ejecutar desde PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

No es necesario abrir manualmente Visual Studio Developer Command Prompt. El instalador inicializa el entorno MSVC automaticamente.

Para diagnostico rapido despues de instalar:

```powershell
python scripts\doctor.py --tests
```

## Linux

El proyecto incluye tambien `setup.sh` para Linux.

```bash
chmod +x setup.sh
./setup.sh
```

El script detecta Conda, CUDA, g++, FFmpeg y FFprobe, compila ambas extensiones CUDA, genera el smoke sintetico y ejecuta el doctor.

La sintaxis Bash se valida en el repositorio; la compatibilidad final de CUDA y compilador depende de la distribucion y del host Linux utilizados.

---

# 18. Orden recomendado de validacion

Despues de instalar:

```text
1. verificar Python
2. verificar PyTorch
3. verificar CUDA disponible
4. verificar NVCC
5. verificar compilador C++
6. compilar raster_cuda
7. compilar gabor_audio_cuda
8. ejecutar pytest
9. ejecutar smoke audiovisual
```

No se recomienda comenzar con un entrenamiento grande antes de completar estas comprobaciones.

---

# 19. Documentacion relacionada

```text
docs/COMMANDS.md
docs/CONFIGURATION.md
docs/PIPELINE.md
docs/TROUBLESHOOTING.md
```
