"""
TODO: Take more inspiration from PyTorch's [losses](https://pytorch.org/docs/stable/nn.html#distance-functions)
E.g.: CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, 
HingeEmbeddingLoss, MultiLabelMarginLoss, HuerLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, 
CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss, CosineSimilarity,

"""

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from neurodiff.compare import DataDependentComparisonMixin, Divergence, Metric


class LP(Metric, DataDependentComparisonMixin):
    """Calculate the L-p divergence between two models."""
    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu', p=2, eps=1e-06):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)
        self.p = p
        self.eps = eps
        self.criterion = nn.PairwiseDistance(p=p, eps=1e-06, keepdim=False)

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        return self.criterion(outputs1, outputs2)


class L1(LP):
    """Calculate the L1 divergence between two models."""
    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device, p=1)


class L2(LP):
    """Calculate the L2 divergence between two models."""
    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device, p=2)


class CrossEntropy(Divergence, DataDependentComparisonMixin):
    """Calculate the cross-entropy between two models."""
    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        return self.criterion(outputs1, outputs2)

class KLDivergence(Divergence, DataDependentComparisonMixin):
    """Calculate the KL divergence between two models."""
    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)
        self.criterion = nn.KLDivLoss

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        log_softmax1 = torch.nn.functional.log_softmax(outputs1, dim=1)
        softmax2 = torch.nn.functional.softmax(outputs2, dim=1)

        return self.criterion(log_softmax1, softmax2)


class SupNorm(Metric, DataDependentComparisonMixin):
    """Calculate the sup-norm between the outputs of model1 and model2."""

    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        outputs1 = model1(inputs)
        softmax1 = torch.nn.functional.softmax(outputs1, dim=1)

        outputs2 = model2(inputs)
        softmax2 = torch.nn.functional.softmax(outputs2, dim=1)

        return torch.max(torch.abs(softmax1 - softmax2)).item()
    
    def compare(self, model1: nn.Module, model2: nn.Module):
        """Compare model1 and model2 using the data-dependent comparison."""
        if not self.dataloader:
            raise ValueError("Dataloader must be provided for data-dependent comparisons.")

        total_sup_norm = 0.0

        with self.eval(model1, model2):
            for inputs, outputs in self.dataloader:
                inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                sup_norm = self.compare_batch(model1, model2, inputs, outputs)    
            
                total_sup_norm = max(total_sup_norm, sup_norm)

        return total_sup_norm



