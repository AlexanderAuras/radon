import typing

import torch
import torch.utils.cpp_extension

_impl = typing.cast(typing.Any, torch.utils.cpp_extension.load(name="radon_impl", sources=["./src/radon.cpp", "./src/forward.cpp", "./src/forward.cu", "./src/backward.cpp", "./src/backward.cu"], extra_cflags=["-fopenmp"]))