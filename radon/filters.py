import abc

import torch
import torch.utils.cpp_extension

from radon.radon_cuda import _cuda_impl



class Filter(abc.ABC):
    @abc.abstractclassmethod
    def _apply(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([])
    
    @abc.abstractclassmethod
    def _cuda_ptr(cls) -> int:
        return 0



class RamLakFilter(Filter):
    @classmethod
    def _apply(cls, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @classmethod
    def _cuda_ptr(cls) -> int:
        return _cuda_impl._ram_lak_filter_ptr



class HannFilter(Filter):
    @classmethod
    def _apply(cls, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @classmethod
    def _cuda_ptr(cls) -> int:
        return _cuda_impl._hann_filter_ptr