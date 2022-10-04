from math import ceil

import torch
import torch.utils.cpp_extension

from radon.radon_impl import _impl



def radon_forward(image: torch.Tensor, thetas: torch.Tensor|None = None, positions: torch.Tensor|None = None) -> torch.Tensor:
    assert image.is_contiguous(), "'image' must be contiugous"
    assert image.dtype == torch.float32, "'image' must be a float32 tensor"
    assert image.dim() == 4, "'image' must have four dimensions [B,C,H,W]"
    assert image.shape[1] == 1, "'image' must have exactly one channel"
    assert image.shape[2] == image.shape[3], "'image' must be square"

    if thetas == None:
        thetas = torch.linspace(0.0, 3.14159265359, 257, device=image.device)[:-1]
    assert thetas.is_contiguous(), "'thetas' must be contiugous"
    assert thetas.dtype == torch.float32, "'thetas' must be a float32 tensor"
    assert thetas.dim() == 1, "'thetas' must have one dimension"
    thetas %= 6.28318530718

    if positions == None:
        M = ceil(image.shape[2]*1.41421356237/2.0)
        positions = torch.arange(-M, M+1, device=image.device).to(torch.float32)
    assert positions.is_contiguous(), "'positions' must be contiugous"
    assert positions.dtype == torch.float32, "'positions' must be a float32 tensor"
    assert positions.dim() == 1, "'positions' must have one dimension"

    assert image.is_cuda == positions.is_cuda and positions.is_cuda == thetas.is_cuda, "All tensors must be on the same device"
    if image.is_cuda:
        return _impl.cuda_forward(image, thetas, positions)
    return _impl.cpu_forward(image, thetas, positions)



def radon_backward(sinogram: torch.Tensor, image_size: int, thetas: torch.Tensor|None = None, positions: torch.Tensor|None = None) -> torch.Tensor:
    assert sinogram.is_contiguous(), "'sinogram' must be contiugous"
    assert sinogram.dtype == torch.float32, "'sinogram' must be a float32 tensor"
    assert sinogram.dim() == 4, "'sinogram' must have four dimensions [B,C,H,W]"
    assert sinogram.shape[1] == 1, "'sinogram' must have exactly one channel"

    if thetas == None:
        thetas = torch.linspace(0.0, 3.14159265359, 257, device=sinogram.device)[:-1]
    assert thetas.is_contiguous(), "'thetas' must be contiugous"
    assert thetas.dtype == torch.float32, "'thetas' must be a float32 tensor"
    assert thetas.dim() == 1, "'thetas' must have one dimension"
    assert thetas.shape[0] == sinogram.shape[2], "size of 'thetas' must match second to last dimension of 'sinogram'"
    thetas %= 6.28318530718

    if positions == None:
        M = ceil(image_size*1.41421356237/2.0)
        positions = torch.arange(-M, M+1, device=sinogram.device).to(torch.float32)
    assert positions.is_contiguous(), "'positions' must be contiugous"
    assert positions.dtype == torch.float32, "'positions' must be a float32 tensor"
    assert positions.dim() == 1, "'positions' must have one dimension"
    assert positions.shape[0] == sinogram.shape[3], "size of 'positions' must match last dimension of 'sinogram'"

    assert sinogram.is_cuda == positions.is_cuda and positions.is_cuda == thetas.is_cuda, "All tensors must be on the same device"
    if sinogram.is_cuda:
        return _impl.cuda_backward(sinogram, thetas, positions, image_size)
    return _impl.cpu_backward(sinogram, thetas, positions, image_size)