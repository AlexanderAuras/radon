import typing

import torch
import torch.utils.cpp_extension

from radon.radon_functional import radon_forward, radon_backward



class RadonForwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, image: torch.Tensor, thetas: typing.Union[torch.Tensor,None] = None, positions: typing.Union[torch.Tensor,None] = None) -> torch.Tensor: #type: ignore
        ctx.image_size = image.shape[2]
        ctx.save_for_backward(thetas, positions)
        return radon_forward(image, thetas, positions)

    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor) -> typing.Tuple[torch.Tensor, None, None]: #type: ignore
        return radon_backward(grad_output.contiguous(), ctx.image_size, *ctx.saved_tensors), None, None



class RadonForward(torch.nn.Module):
    def __init__(self, thetas: typing.Union[torch.Tensor,None] = None, positions: typing.Union[torch.Tensor,None] = None) -> None:
        super().__init__()
        self.thetas = torch.nn.parameter.Parameter(thetas, requires_grad=False) if thetas!=None else None
        self.positions = torch.nn.parameter.Parameter(positions, requires_grad=False) if positions!=None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RadonForwardFunc.apply(x, self.thetas, self.positions)



class RadonBackwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, sinogram: torch.Tensor, image_size: int, thetas: typing.Union[torch.Tensor,None] = None, positions: typing.Union[torch.Tensor,None] = None) -> torch.Tensor: #type: ignore
        ctx.image_size = image_size
        ctx.save_for_backward(thetas, positions)
        return radon_backward(sinogram, image_size, thetas, positions)

    @staticmethod
    def backward(ctx: typing.Any, grad_output: torch.Tensor) -> typing.Tuple[torch.Tensor, None, None, None]: #type: ignore
        return radon_forward(grad_output.contiguous(), *ctx.saved_tensors), None, None, None



class RadonBackward(torch.nn.Module):
    def __init__(self, image_size: int, thetas: typing.Union[torch.Tensor,None] = None, positions: typing.Union[torch.Tensor,None] = None) -> None:
        super().__init__()
        self.image_size = torch.nn.parameter.Parameter(torch.tensor(image_size), requires_grad=False)
        self.thetas = torch.nn.parameter.Parameter(thetas, requires_grad=False) if thetas!=None else None
        self.positions = torch.nn.parameter.Parameter(positions, requires_grad=False) if positions!=None else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RadonBackwardFunc.apply(x, self.image_size, self.thetas, self.positions)