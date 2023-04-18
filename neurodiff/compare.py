from contextlib import contextmanager
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class Comparison:
    """The base class for all neural network comparisons."""
    type_: Literal['metric', 'divergence', 'misc'] = 'misc'

    def __init__(self, reduction='mean', device='cpu'):
        self.reduction = reduction
        self.device = device

        if reduction != 'mean':       
            raise NotImplementedError

    def compare(self, model1: nn.Module, model2: nn.Module):
        """Compare model1 and model2."""
        raise NotImplementedError

    def __call__(self, model1: nn.Module, model2: nn.Module):
        """Compare model1 and model2."""
        return self.compare(model1, model2)
    
    @contextmanager
    def eval(self, model1: nn.Module, model2: nn.Module):
        """Context manager that moves models to the device and sets them to eval mode."""

        model1.to(self.device)
        model2.to(self.device)

        model1_was_training = model1.training
        model2_was_training = model2.training

        model1.eval()
        model2.eval()

        with torch.no_grad():
            try:
                yield
            finally:
                if model1_was_training:
                    model1.train()
                if model2_was_training:
                    model2.train()
    
    @property
    def is_symmetric(self):
        """Return True if the comparison is symmetric."""
        return self.type_ == 'metric'

    def compare_symmetric(self, model1: nn.Module, model2: nn.Module):
        """Return a symmetrized version of the comparison."""
        if self.is_symmetric:
            return self.compare(model1, model2)
        else:
            raise (self.compare(model1, model2) + self.compare(model2, model1)) / 2