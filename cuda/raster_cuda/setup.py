import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    try:
        value_int = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} debe ser un entero. Ejemplo: 128, 256, 512") from exc
    if value_int <= 0:
        raise ValueError(f"{name} debe ser mayor que 0")
    return value_int


raster_batch_size = _get_int_env("RASTER_BATCH_SIZE", 256)

# Build flags focused on release speed.
# --use_fast_math can slightly change floating point results, but it matches the
# current optimized rasterizer strategy.
cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-Xptxas=-O3",
    f"-DRASTER_BATCH_SIZE={raster_batch_size}",
]

# Optional extra flags for experiments, for example:
#   set RASTER_EXTRA_NVCC_FLAGS=--maxrregcount=96
extra_nvcc = os.environ.get("RASTER_EXTRA_NVCC_FLAGS", "").strip()
if extra_nvcc:
    nvcc_flags.extend(extra_nvcc.split())

print(f"[setup] RASTER_BATCH_SIZE={raster_batch_size}")
if extra_nvcc:
    print(f"[setup] extra nvcc flags={extra_nvcc}")

setup(
    name="raster_cuda",
    ext_modules=[
        CUDAExtension(
            name="raster_cuda",
            sources=[
                "raster_cuda.cpp",
                "raster_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
