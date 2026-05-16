from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="laplacian_1d_cuda",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="laplacian_1d_cuda",
            sources=[
                os.path.join(this_dir, "src/wrapper.cpp"),
                os.path.join(this_dir, "src/laplacian_1d_cuda_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--use_fast_math",
                    "-lineinfo",
                    "-Xptxas=-v",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
