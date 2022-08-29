import typing

import torch
import torch.utils.cpp_extension

_cuda_impl = typing.cast(typing.Any, torch.utils.cpp_extension.load(name="radon_impl", extra_cuda_cflags=["-G"], sources=["./src/radon.cpp", "./src/forward.cu", "./src/backward.cu", "./src/inverse.cu", "./src/filters.cu"]))