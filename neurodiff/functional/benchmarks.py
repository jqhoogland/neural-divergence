"""
Comparisons that compare the performance of two models on a dataset (at the level of loss).
"""

from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from neurodiff.compare import Comparison


class LossComparison(Comparison):
    """The base class for all neural network comparisons that compare models at the level of loss.
    """

    def __init__(self, dataloader: DataLoader, loss: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader = dataloader
        self.loss = loss

    def score(self, model: nn.Module):
        """Score model on the dataset."""
        loss = torch.Tensor(0.0, device=self.device)

        for inputs, outputs in self.dataloader:
            inputs, outputs = inputs.to(self.device), outputs.to(self.device)
            loss += self.loss(model(inputs), outputs)
        
        loss /= len(self.dataloader)

        return loss
    
    def compare(self, model1: nn.Module, model2: nn.Module):
        """Compare model1 and model2."""
        with self.eval(model1, model2):
            loss1 = self.score(model1)
            loss2 = self.score(model2)

        return loss1 - loss2

    def compare_symmetric(self, model1: nn.Module, model2: nn.Module):
        """Return a symmetrized version of the comparison."""
        return torch.abs(self.compare(model1, model2))