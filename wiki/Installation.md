# Installation

## Requirements

Python ≥ 3.10. For training, an NVIDIA GPU with CUDA and a C++ compiler
compatible with your PyTorch build. FFmpeg and FFprobe must be on `PATH` for
video and audio extraction; 7-Zip is optional and only used to losslessly
compress the final packaged models.

The CPU-only path — tests, checkpoint inspection, metrics on existing frames —
needs no GPU.

## Environment

```bash
git clone https://github.com/JLeandroJM01/MBB-GS_MOVING_GAUSSIANS.git
cd MBB-GS_MOVING_GAUSSIANS

conda create -n mbb-gs python=3.10 -y
conda activate mbb-gs
python -m pip install --upgrade pip
```

## Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` lists direct dependencies with lower bounds, so it installs
on any platform. For a GPU environment, install PyTorch from the CUDA index
first so that pip resolves the right wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

**`pip install -e .` is not optional.** The scripts import `gs2d_video` and
`gs2d_gabor` as installed packages; without the editable install they fail with
`ModuleNotFoundError: No module named 'gs2d_video'`. If you run on a cluster,
run it once inside the environment the jobs activate.

The exact environment used on the Khipu cluster to produce the reported numbers
is frozen in `requirements-tesis-khipu.txt`. Those wheels carry the `+cu126`
local tag, so that file only installs against the PyTorch CUDA 12.6 index:

```bash
pip install -r requirements-tesis-khipu.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

## Verifying PyTorch and CUDA

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda runtime:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

The CUDA version PyTorch was built against, the NVIDIA driver, and the toolkit
used to compile the extensions all have to be mutually compatible. Resolve this
before compiling anything.

## Building the CUDA extensions

There are two, and the compiled binaries are deliberately not versioned — they
are tied to a specific Python ABI and CUDA build, so they must be produced in
each environment.

### Video rasteriser

```bash
cd cuda/raster_cuda
python setup.py build_ext --inplace
cd ../..
```

This produces `raster_cuda*.so` on Linux or `raster_cuda.cp310-win_amd64.pyd`
on Windows. Verify:

```bash
python -c "import torch, sys; sys.path.insert(0, 'cuda/raster_cuda'); import raster_cuda; print('raster_cuda OK')"
```

Import `torch` before the extension: on Windows that is what loads the required
DLLs.

The Python wrapper looks for the extension in `cuda/raster_cuda` relative to the
repository root, or wherever the `RUTA_RASTER_CUDA` environment variable points.

#### `RASTER_BATCH_SIZE`

A compile-time constant, default 256, that sets how many Gaussians a tile loads
into shared memory per batch; it sizes the `__shared__` arrays in the kernel.
Changing it requires recompiling:

```bash
RASTER_BATCH_SIZE=256 python setup.py build_ext --inplace
```

### Gabor audio

Only needed for the audio extension:

```bash
cd cuda/gabor_audio_cuda
python setup.py build_ext --inplace
cd ../..

python -c "import torch, sys; sys.path.insert(0, 'cuda/gabor_audio_cuda'); import gabor_audio_cuda; print('gabor_audio_cuda OK')"
```

It is located the same way, via `RUTA_GABOR_AUDIO_CUDA` if set. If the extension
is missing, the audio code falls back to a dense PyTorch implementation that is
correct but far too slow for real training — see [Gabor Audio](Gabor-Audio).

### Multiple GPU architectures

On a cluster with heterogeneous nodes, name the architectures explicitly so the
binary runs everywhere it is scheduled:

```bash
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"
```

Adapt the list to the actual hardware.

## Checking the installation

```bash
pytest -q
```

Eight tests, CPU only, no trained model required. They do not exercise the CUDA
rasteriser — for that, run a short training with a small configuration.

If something fails, see [Troubleshooting](Troubleshooting).
