import os
import sys
import typing

import torch
import torch.utils.cpp_extension

if sys.platform == "win32":
    _compiler = os.environ.get("CXX", "cl")
else:
    _compiler = os.environ.get("CXX", "g++")
_sources = ["./src/radon.cpp", "./src/forward.cpp", "./src/backward.cpp"]
_cflags = [{"g++": "-fopenmp", "cl": "/openmp"}[_compiler]]
if torch.cuda.is_available():
    _sources.extend(["./src/forward.cu", "./src/backward.cu"])
    _cflags.extend([{"g++": "-DRADON_CUDA_AVAILABLE", "cl": "/DRADON_CUDA_AVAILABLE"}[_compiler]])
_impl = typing.cast(typing.Any, torch.utils.cpp_extension.load(name="radon_impl", sources=_sources, extra_cflags=_cflags))
del _sources
del _cflags
del _compiler