import typing

import torch
import torch.utils.cpp_extension

from radon.radon_functional import radon_forward, radon_backward



class RadonForwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, image: torch.Tensor, thetas: torch.Tensor|None = None, positions: torch.Tensor|None = None) -> torch.Tensor:
        ctx.thetas = thetas
        ctx.positions = positions
        ctx.image_size = image.shape[2]
        args = [image]
        if thetas != None:
            args.append(thetas)
        if positions != None:
            args.append(positions)
        return radon_forward(*args)

    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        args = [grad_output.contiguous(), ctx.image_size]
        if ctx.thetas != None:
            args.append(ctx.thetas)
        if ctx.positions != None:
            args.append(ctx.positions)
        return radon_backward(*args), None, None



class RadonForward(torch.nn.Module):
    def __init__(self, thetas: torch.Tensor|None = None, positions: torch.Tensor|None = None) -> None:
        super().__init__()
        self.thetas = torch.nn.parameter.Parameter(thetas, requires_grad=False) if thetas!=None else None
        self.positions = torch.nn.parameter.Parameter(positions, requires_grad=False) if positions!=None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RadonForwardFunc.apply(x, self.thetas, self.positions)



class RadonBackwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, sinogram: torch.Tensor, image_size: int, thetas: torch.Tensor|None = None, positions: torch.Tensor|None = None) -> torch.Tensor:
        ctx.image_size = image_size
        ctx.thetas = thetas
        ctx.positions = positions
        args = [sinogram, image_size]
        if thetas != None:
            args.append(thetas)
        if positions != None:
            args.append(positions)
        return radon_backward(*args)

    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        args = [grad_output.contiguous()]
        if ctx.thetas != None:
            args.append(ctx.thetas)
        if ctx.positions != None:
            args.append(ctx.positions)
        return radon_forward(*args), None, None, None



class RadonBackward(torch.nn.Module):
    def __init__(self, image_size: int, thetas: torch.Tensor|None = None, positions: torch.Tensor|None = None) -> None:
        super().__init__()
        self.image_size = torch.nn.parameter.Parameter(torch.tensor(image_size), requires_grad=False)
        self.thetas = torch.nn.parameter.Parameter(thetas, requires_grad=False) if thetas!=None else None
        self.positions = torch.nn.parameter.Parameter(positions, requires_grad=False) if positions!=None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RadonBackwardFunc.apply(x, self.image_size, self.thetas, self.positions)