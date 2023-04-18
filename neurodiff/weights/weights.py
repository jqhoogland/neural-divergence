"""
TODO: These need to be carefully checked. 
It's very possible that the differences are being accumulated incorrectly across layers.
"""

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from neurodiff.compare import Comparison


class WeightsComparison(Comparison):
    """
    A class for comparisons between the weights of two models.
    """
    def __init__(self, *args, include_bias=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_bias = include_bias

    def compare_weight(self, weight1: Tensor, weight2: Tensor):
        raise NotImplementedError

    def compare_layer(self, layer1: nn.Module, layer2: nn.Module):
        """Compare layer1 and layer2."""
        layer_type = type(layer1)

        if layer_type != type(layer2):
            raise TypeError("Layers must be of the same type.")

        diff = self.compare_weight(layer1.weight, layer2.weight)

        if self.include_bias and hasattr(layer1, 'bias'):
            diff += self.compare_weight(layer1.bias, layer2.bias)

        return diff
    
    def numel(self, layer: nn.Module):       
        numel = layer.weight.numel()

        if self.include_bias and hasattr(layer, 'bias'):
            numel += layer.bias.numel()

        return numel
    
    def norm(self, model: nn.Module):
        """Calculate the norm of the weights of a model."""
        total_norm = 0.0

        for layer in model.modules():
            try: 
                total_norm += torch.norm(layer.weight)

                if self.include_bias and hasattr(layer, 'bias'):
                    total_norm += torch.norm(layer.bias)

            except AttributeError:
                pass

        return total_norm 

    def compare(self, model1: nn.Module, model2: nn.Module):
        """Compare model1 and model2 using the data-dependent comparison."""
        total_diff = 0.0
        num_params = 0

        with self.eval(model1, model2):
            for layer1, layer2 in zip(model1.modules(), model2.modules()):
                try: 
                    total_diff += self.compare_layer(layer1, layer2)
                    num_params += self.numel(layer1)
                except AttributeError:  # If layer does not have `weight`
                    pass
                
        if self.reduction == 'mean':
            total_diff /= num_params

        return total_diff 
    


class LPWeights(WeightsComparison):
    def __init__(self, *args, p: int = 2, **kwargs):
        # TODO: Get this to work with inf norm
        kwargs["reduction"] = "sum"
        super().__init__(*args, **kwargs)
        self.p = p

    def compare_weight(self, weight1: Tensor, weight2: Tensor):
        return torch.sum(torch.abs(weight1 - weight2) ** self.p)
    
    def compare(self, model1: nn.Module, model2: nn.Module):
        return super().compare(model1, model2) ** (1 / self.p)  
    

class L1Weights(LPWeights):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, p=1, **kwargs)


class L2Weights(LPWeights):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, p=2, **kwargs)


class CosineSimWeights(WeightsComparison):
    def __init__(self, *args, **kwargs):
        kwargs['reduction'] = 'sum'
        super().__init__(*args, **kwargs)

    def compare_weight(self, weight1: Tensor, weight2: Tensor):
        return (weight1 * weight2).sum() 

    def compare(self, model1: nn.Module, model2: nn.Module):
        return super().compare(model1, model2) / (self.norm(model1) * self.norm(model2))