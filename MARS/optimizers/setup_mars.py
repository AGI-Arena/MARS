"""
Fused MARS setup script.
Temporarily implement for AdamW variant only.

Usage: 
    run "python setup_mars.py install" to build and install the fused MARS optimizer.
    Then import mars_fused in your training script.

For fused version, pass the argument "fused=True" to MARS optimizer.
For multi-tensor implementation, pass the argument "foreach=True" to MARS optimizer.

Fused multi-tensor MARS requires further improvements and optimizations.

"""

import os
import sys
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.cuda import is_available

build_cuda_ext = is_available() or os.getenv('FORCE_CUDA', '0') == '1'

cuda_extension = None
if "--unfused" in sys.argv:
    print("Building unfused version of MARS")
    sys.argv.remove("--unfused")
elif build_cuda_ext:
    cuda_extension = CUDAExtension(
        'fused_mars', 
        sources=['fused_mars/pybind_mars.cpp', './fused_mars/fused_mars_kernel.cu', './fused_mars/multi_tensor_mars_kernel.cu']
    )

setup(
    name='mars',
    python_requires='>=3.8',
    version='0.0.1',
    install_requires=['torch'],
    py_modules=['mars'],
    description=(
        'MARS: Unleashing the Power of '
        'Variance Reduction for Training Large Models'
    ),
    author=(
        'Yuan, Huizhuo and Liu, Yifeng and Wu, Shuang and '
        'Zhou, Xun and Gu, Quanquan'
    ),
    ext_modules=[cuda_extension] if cuda_extension is not None else [],
    cmdclass={'build_ext': BuildExtension} if build_cuda_ext else {},
)