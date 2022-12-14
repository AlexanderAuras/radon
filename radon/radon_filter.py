import typing

import torch



def radon_filter(sinogram: torch.Tensor, filter: typing.Callable[[torch.Tensor, typing.Any], torch.Tensor], params: typing.Any = None) -> torch.Tensor:
    assert sinogram.dtype == torch.float32, "'sinogram' must be a float32 tensor"
    assert sinogram.dim() == 4, "'sinogram' must have four dimensions [B,C,H,W]"
    assert sinogram.shape[1] == 1, "'sinogram' must have exactly one channel"
    #return typing.cast(torch.Tensor, torch.fft.ifft(torch.fft.ifftshift(filter(torch.fft.fftshift(torch.fft.fft(sinogram, dim=3, norm="forward"), dim=3), params), dim=3), dim=3, norm="forward")).real.contiguous()
    pos_length = sinogram.shape[3]
    return typing.cast(torch.Tensor, torch.fft.irfft(filter(torch.fft.rfft(sinogram, dim=3, norm="forward"), params), n=pos_length, dim=3, norm="forward")).real.contiguous()



def ram_lak_filter(sinogram: torch.Tensor, _: typing.Any) -> torch.Tensor:
    #filter = torch.abs(torch.arange(0, sinogram.shape[3], device=sinogram.device).to(torch.float32)-sinogram.shape[3]//2)
    filter = torch.abs(torch.arange(0, sinogram.shape[3], device=sinogram.device).to(torch.float32))
    #filter[sinogram.shape[3]//2] = 0.25
    filter[0] = 0.25
    return sinogram*filter.reshape(1,1,1,-1)



class RadonFilter(torch.nn.Module):
    def __init__(self, filter: typing.Callable[[torch.Tensor, typing.Any], torch.Tensor], params: typing.Any = None) -> None:
        super().__init__()
        self.filter = filter
        self.params = params
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return radon_filter(x, self.filter, self.params)