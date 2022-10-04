from turtle import shape
import typing

import torch



def radon_filter(sinogram: torch.Tensor, filter: typing.Callable[[torch.Tensor], torch.Tensor], params: typing.Any) -> torch.Tensor:
    assert sinogram.dtype == torch.float32, "'sinogram' must be a float32 tensor"
    assert sinogram.dim() == 4, "'sinogram' must have four dimensions [B,C,H,W]"
    assert sinogram.shape[1] == 1, "'sinogram' must have exactly one channel"
    return typing.cast(torch.Tensor, torch.fft.ifft(torch.fft.ifftshift(filter(torch.fft.fftshift(torch.fft.fft(sinogram, dim=3 ,norm = 'forward'), dim=3), params), dim=3), dim=3,norm = 'forward')).real



def ram_lak_filter(sinogram: torch.Tensor, _: typing.Any) -> torch.Tensor:
    filter = torch.abs(torch.arange(0,sinogram.shape[3])-sinogram.shape[3]//2)
    filter[0] = 1/4
    return sinogram*filter.reshape(1,1,1,-1)