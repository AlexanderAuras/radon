import abc

import torch
import torch.utils.cpp_extension

from radon.radon_impl import _impl



class Filter(abc.ABC):
    @abc.abstractclassmethod
    def _apply(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([])
    
    @abc.abstractclassmethod
    def _cuda_id(cls) -> int:
        return -1



class RamLakFilter(Filter):
    @classmethod
    def _apply(cls, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @classmethod
    def _cuda_id(cls) -> int:
        return _impl._RAM_LAK_FILTER_ID



class HannFilter(Filter):
    @classmethod
    def _apply(cls, x: torch.Tensor) -> torch.Tensor:
        return x
    
    @classmethod
    def _cuda_id(cls) -> int:
        return _impl._HANN_FILTER_ID