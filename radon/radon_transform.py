import typing

import torch
import torch.utils.cpp_extension

from radon.radon_cuda import *



class RadonTransformFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, image: torch.Tensor, angles: torch.Tensor|None = None, positions: torch.Tensor|None = None) -> torch.Tensor:
        ctx.angles = angles
        ctx.positions = positions
        ctx.image_size = image.shape[2]
        args = [image]
        if angles != None:
            args.append(angles)
        if positions != None:
            args.append(positions)
        return radon_forward(*args)

    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        args = [grad_output.contiguous(), ctx.image_size]
        if ctx.angles != None:
            args.append(ctx.angles)
        if ctx.positions != None:
            args.append(ctx.positions)
        return radon_backward(*args), None, None



class RadonTransform(torch.nn.Module):
    def __init__(self, angles: torch.Tensor|None = None, positions: torch.Tensor|None = None) -> None:
        super().__init__()
        self.angles = angles
        self.positions = positions
    
    def forward(self, x: torch.Tensor) -> None:
        return RadonTransformFunc.apply(x, self.angles, self.positions)