# Troubleshooting

## `ModuleNotFoundError: No module named 'gs2d_video'`

The package is not installed in the active environment. From the repository
root:

```bash
pip install -e .
```

This is the most common failure on a cluster, where the job activates a conda
environment that was created before the package existed. The scripts import
`gs2d_video` as an installed package and do not patch `sys.path`.

## `ModuleNotFoundError: No module named 'raster_cuda'`

The CUDA extension was not compiled, is not where the wrapper looks, or was
built for a different Python environment.

```bash
cd cuda/raster_cuda
python setup.py build_ext --inplace
cd ../..

python -c "import torch, sys; sys.path.insert(0, 'cuda/raster_cuda'); import raster_cuda; print('OK')"
```

Import `torch` before the extension — on Windows that is what loads the DLLs it
depends on.

The wrapper looks in `cuda/raster_cuda` relative to the repository root, or at
`RUTA_RASTER_CUDA` if that variable is set.

## The binary exists but will not import

A compiled extension is bound to a specific Python version and ABI. A file named
`raster_cuda.cp310-win_amd64.pyd` will only load under CPython 3.10 on Windows
x64, built against a compatible PyTorch. Copying a binary from another machine
or another environment does not work.

Check what you actually have:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version
nvidia-smi
```

## CUDA compilation fails

Verify the CUDA Toolkit, `nvcc`, the C++ compiler, PyTorch and the GPU
architecture are mutually compatible. On Windows, PyTorch CUDA extensions need a
matching MSVC toolchain. On Slurm, check that the loaded modules match the
environment the job actually runs in — loading `cuda/12.8` while PyTorch was
built for 12.6 is a common mismatch.

For heterogeneous nodes, name the architectures explicitly:

```bash
export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"
```

## `torch.cuda.is_available() == False`

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Resolve this before compiling or training. A CPU-only PyTorch build will never
see the GPU no matter how the extension is compiled.

## Out of memory

First determine **which** memory ran out. Keeping frames on the CPU reduces VRAM
at the cost of system RAM, so the fix is opposite in each case.

For VRAM, the recommended configuration for large trainings is:

```json
"frames_en_cpu": true,
"frames_en_gpu_uint8": false,
"evitar_render_completo_en_train": true,
"usar_metricas_streaming": true
```

If it still does not fit, reduce `n_gaussianas_inicial`, the resolution, the
number of frames, `sub_batch_frames`, or the temporal degrees — roughly in that
order of effect.

Remember that a clip at 720p in float32 costs about 11 MB per frame just to
store, before the model exists. See [Training](Training).

## LPIPS is `null` in the metrics

LPIPS downloads AlexNet weights on first use, which fails on a compute node
without internet. Compute it afterwards from the saved frames:

```bash
python scripts/metrics/lpips_post_hoc.py --exp outputs/mi_experimento
```

## `ffmpeg` not found

FFmpeg and FFprobe must be on `PATH`. They are used for frame extraction, audio
extraction and the final mux in the audiovisual pipeline.

## `7z` not found

Only used to losslessly compress the final packages, and optional. On Windows
the pipeline looks for the default install location; elsewhere, `7z` should be
on `PATH`.

## The output folder already exists

Training refuses to overwrite by default. Either change `nombre_experimento`,
pass `--nombre-experimento NAME`, or set `"sobreescribir_salida": true` to reuse
the folder. Reusing does not delete previous contents: new files overwrite, old
ones remain, which can leave a confusing mix.

## `usar_loss_cuda` rejected

```
usar_loss_cuda=true only supports tipo_loss baseline/l1 with lambda_dssim=0.0
```

The fused kernel implements only L1 and MSE. Anything involving motion, edges,
temporal terms or DSSIM must run with `usar_loss_cuda: false`. See
[Loss Functions](Loss-Functions).

## pytest picks up old CUDA tests

It should not: `pyproject.toml` sets `testpaths = ["tests"]`. If you invoke
pytest with an explicit path that reaches `cuda/raster_cuda/tests/`, you will
hit historical benchmarks that import module names from earlier versions of the
code. They are not maintained and are not a health signal for the repository.

## Gabor CUDA and PyTorch differ slightly

Small numerical differences between the CUDA kernel and the dense PyTorch
fallback are expected: different accumulation orders, and `--use_fast_math` in
the build. What must hold is that the analytic gradients match autograd, which
is what `scripts/audio/test_gradientes_gabor.py` and the test suite check.
