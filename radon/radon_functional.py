import typing

import torch
import torch.utils.cpp_extension

from radon.radon_cuda import _cuda_impl
from radon.filters import Filter, RamLakFilter



def radon_forward(image: torch.Tensor, angles: torch.Tensor = torch.linspace(0.0, 3.14159265359, 256, device="cuda"), positions: torch.Tensor|None = None) -> torch.Tensor:
    assert image.is_cuda, "'image' must be a CUDA tensor"
    assert image.is_contiguous(), "'image' must be contigous"
    assert image.dtype == torch.float32 or image.dtype == torch.float64, "'image' must be a float tensor"
    assert image.dim() == 4, "'image' must have four dimensions [B,C,H,W]"
    assert image.shape[1] == 1, "'image' must have exactly one channel"
    assert image.shape[2] == image.shape[3], "'image' must be square"

    assert angles.is_cuda, "'angles' must be a CUDA tensor"
    assert angles.is_contiguous(), "'angles' must be contigous"
    assert angles.dtype == torch.float32 or angles.dtype == torch.float64, "'angles' must be a float tensor"
    assert angles.dim() == 1, "'angles' must have one dimension"
    angles %= 6.28318530718

    if positions == None:
        positions = torch.arange(0.0, image.shape[2], device="cuda")
    assert positions.is_cuda, "'positions' must be a CUDA tensor"
    assert positions.is_contiguous(), "'positions' must be contigous"
    assert positions.dtype == torch.float32 or positions.dtype == torch.float64, "'positions' must be a float tensor"
    assert positions.dim() == 1, "'positions' must have one dimension"
    assert torch.min(positions) >= 0.0 and torch.max(positions) <= image.shape[2], "all values of 'positions' must be in [0,image.shape[2]]"
    
    return _cuda_impl.forward(image, angles, positions)



def radon_backward(sinogram: torch.Tensor, image_size: int, angles: torch.Tensor = torch.linspace(0.0, 3.14159265359, 256, device="cuda"), positions: torch.Tensor|None = None, filter: typing.Type[Filter] = RamLakFilter) -> torch.Tensor:
    assert sinogram.is_cuda, "'sinogram' must be a CUDA tensor"
    assert sinogram.is_contiguous(), "'sinogram' must be contigous"
    assert sinogram.dtype == torch.float32 or sinogram.dtype == torch.float64, "'sinogram' must be a float tensor"
    assert sinogram.dim() == 4, "'sinogram' must have four dimensions [B,C,H,W]"
    assert sinogram.shape[1] == 1, "'sinogram' must have exactly one channel"

    assert angles.is_cuda, "'angles' must be a CUDA tensor"
    assert angles.is_contiguous(), "'angles' must be contigous"
    assert angles.dtype == torch.float32 or angles.dtype == torch.float64, "'angles' must be a float tensor"
    assert angles.dim() == 1, "'angles' must have one dimension"
    assert angles.shape[0] == sinogram.shape[2], "size of 'angles' must match second to last dimension of 'sinogram'"
    angles %= 6.28318530718

    if positions == None:
        positions = torch.arange(0.0, image_size, device="cuda")
    assert positions.is_cuda, "'positions' must be a CUDA tensor"
    assert positions.is_contiguous(), "'positions' must be contigous"
    assert positions.dtype == torch.float32 or positions.dtype == torch.float64, "'positions' must be a float tensor"
    assert positions.dim() == 1, "'positions' must have one dimension"
    assert torch.min(positions) >= 0.0 and torch.max(positions) <= sinogram.shape[2], "all values of 'positions' must be in [0,image.shape[2]]"
    assert positions.shape[0] == sinogram.shape[3], "size of 'positions' must match last dimension of 'sinogram'"
    
    return _cuda_impl.backward(sinogram, angles, positions, image_size, filter._cuda_ptr())



def radon_inverse(sinogram: torch.Tensor, image_size: int, angles: torch.Tensor = torch.linspace(0.0, 3.14159265359, 256, device="cuda"), positions: torch.Tensor|None = None) -> torch.Tensor:
    assert sinogram.is_cuda, "'sinogram' must be a CUDA tensor"
    assert sinogram.is_contiguous(), "'sinogram' must be contigous"
    assert sinogram.dtype == torch.float32 or sinogram.dtype == torch.float64, "'sinogram' must be a float tensor"
    assert sinogram.dim() == 4, "'sinogram' must have four dimensions [B,C,H,W]"
    assert sinogram.shape[1] == 1, "'sinogram' must have exactly one channel"

    assert angles.is_cuda, "'angles' must be a CUDA tensor"
    assert angles.is_contiguous(), "'angles' must be contigous"
    assert angles.dtype == torch.float32 or angles.dtype == torch.float64, "'angles' must be a float tensor"
    assert angles.dim() == 1, "'angles' must have one dimension"
    assert angles.shape[0] == sinogram.shape[2], "size of 'angles' must match second to last dimension of 'sinogram'"
    angles %= 6.28318530718

    if positions == None:
        positions = torch.arange(0.0, image_size, device="cuda")
    assert positions.is_cuda, "'positions' must be a CUDA tensor"
    assert positions.is_contiguous(), "'positions' must be contigous"
    assert positions.dtype == torch.float32 or positions.dtype == torch.float64, "'positions' must be a float tensor"
    assert positions.dim() == 1, "'positions' must have one dimension"
    assert torch.min(positions) >= 0.0 and torch.max(positions) <= sinogram.shape[2], "all values of 'positions' must be in [0,image.shape[2]]"
    assert positions.shape[0] == sinogram.shape[3], "size of 'positions' must match last dimension of 'sinogram'"
    
    return _cuda_impl.inverse(sinogram, angles, positions, image_size)