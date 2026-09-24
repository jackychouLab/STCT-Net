from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="tdc_deform_conv_cuda",
    ext_modules=[
        CUDAExtension(
            name="tdc_deform_conv_cuda",
            sources=[
                "deform_conv_3d_cuda.cpp",
                "deform_conv_3d_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
