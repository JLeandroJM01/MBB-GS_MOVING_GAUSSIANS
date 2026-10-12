import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-Xptxas=-O3",
]

setup(
    name="gabor_audio_cuda",
    ext_modules=[
        CUDAExtension(
            name="gabor_audio_cuda",
            sources=[
                "gabor_audio_cuda.cpp",
                "gabor_audio_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
