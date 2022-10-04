# Radon
This repository contains a Python/C++ implementation of the radon transform, based on pytorch and its JIT-C++-compilation feature.

## Features:
 - Forward and backward transforms
 - Filtering with different filters
 - Acceleration via CUDA/OpenMP
 - Differentiability

## Roadmap:
 - Fix of CUDA backward bug
 - Find reason for difference to `torch_radon`
 - Filtering as `autograd.function` and `nn.module`
 - Test differentiability w.r.t. filter parameters
 - PyPI/conda package
 - Support for different floating types
 - Remove duplicate code
 - Better detection of CUDA block size
 - More filter functions

## Known bugs:
 - On unix: `/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’`: caused by weird nvcc implementation, see https://github.com/NVIDIA/nccl/issues/650#issuecomment-1145173577