# Radon
This repository contains a Python/C++ implementation of the radon transform, based on pytorch and its JIT-C++-compilation feature.

## Features:
 - Forward and backward transforms
 - Filtering with different filters
 - Acceleration via CUDA/OpenMP
 - Differentiability
 - Generation of radon transform matrices

## Roadmap:
 - Find reason for difference to `torch_radon`
 - Find reason for pattern in cube reconstruction which does not exist with `torch_radon`
 - PyPI/conda package
 - Support for different floating types
 - Remove duplicate code
 - Better detection of CUDA block size
 - More filter functions
 - Documentation
 - Normalization
 - ~~Fix CUDA backward bug~~
 - ~~Generate transform matrix~~
 - ~~Filtering as `nn.module`~~

## Known bugs:
 - On unix: `/usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’`: caused by weird nvcc implementation, see https://github.com/NVIDIA/nccl/issues/650#issuecomment-1145173577