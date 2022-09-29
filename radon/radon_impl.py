import typing

import torch
import torch.utils.cpp_extension

if torch.cuda.is_available():
    sources = ["./src/radon.cpp", "./src/forward.cpp", "./src/forward.cu", "./src/backward.cpp", "./src/backward.cu"]
else:
    sources = ["./src/radon.cpp", "./src/forward.cpp", "./src/backward.cpp"]
_impl = typing.cast(typing.Any, torch.utils.cpp_extension.load(name="radon_impl", sources=sources, extra_cflags=["-fopenmp", "/openmp"]))
del sources