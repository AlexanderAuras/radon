import contextlib
import importlib.resources
import os
import sys
import typing

import torch
import torch.utils.cpp_extension

if sys.platform == "win32":
    _compiler = os.path.basename(os.environ.get("CXX", "cl"))
else:
    _compiler = os.path.basename(os.environ.get("CXX", "g++"))
_sources = ["radon.cpp", "forward.cpp", "backward.cpp", "matrix.cpp"]#, "matrix_bwd.cpp", "matrix_fwd.cpp"]
_cflags = [{"g++": "-fopenmp", "cl": "/openmp"}[_compiler]]
_ldflags = None
#_ldflags = ['/LIBPATH:C:\\Users\\danil\\AppData\\Local\\Programs\\Python\\Python39\\libs'] 
if torch.cuda.is_available():
    _sources.extend(["forward.cu", "backward.cu", "matrix.cu"])
    _cflags.extend([{"g++": "-DRADON_CUDA_AVAILABLE", "cl": "/DRADON_CUDA_AVAILABLE"}[_compiler]])
#_tmp_file_ctx_managers = list(map(lambda x: importlib.resources.path("radon.cpp", x), _sources))
#with contextlib.ExitStack() as stack:
#    tmp_sources = []
#    for ctx_manager in _tmp_file_ctx_managers:
#        tmp_sources.append(stack.enter_context(ctx_manager).resolve())
#    _impl = typing.cast(typing.Any, torch.utils.cpp_extension.load(name="radon_impl", sources=tmp_sources, extra_cflags=_cflags))
#del _tmp_file_ctx_managers
_impl = typing.cast(typing.Any, torch.utils.cpp_extension.load(name="radon_impl", sources=list(map(lambda x: os.path.join(os.path.dirname(__file__), "cpp", x), _sources)), extra_cflags=_cflags, extra_ldflags=_ldflags))
del _sources
del _cflags
del _compiler