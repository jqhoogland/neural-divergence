"""
Comparisons that directly compare the outputs of two models.

TODO: Take more inspiration from PyTorch's [losses](https://pytorch.org/docs/stable/nn.html#distance-functions)
E.g.: CTCLoss, NLLLoss, PoissonNLLLoss, GaussianNLLLoss, BCELoss, BCEWithLogitsLoss, MarginRankingLoss, 
HingeEmbeddingLoss, MultiLabelMarginLoss, HuerLoss, SmoothL1Loss, SoftMarginLoss, MultiLabelSoftMarginLoss, 
CosineEmbeddingLoss, MultiMarginLoss, TripletMarginLoss, TripletMarginWithDistanceLoss, CosineSimilarity,

"""
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from neurodiff.compare import Comparison


class OutputComparison(Comparison):
    """
    A class for data-dependent comparisons over outputs. 
    """

    def __init__(self, dataloader: DataLoader, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader = dataloader

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        """Compare model1 and model2 on a single batch. Only required for data-dependent comparisons."""
        raise NotImplementedError
    
    def compare(self, model1: nn.Module, model2: nn.Module):
        """Compare model1 and model2 using the data-dependent comparison."""
        if not self.dataloader:
            raise ValueError("Dataloader must be provided for data-dependent comparisons.")

        total_diff = 0.0

        with self.eval(model1, model2):
            for inputs, outputs in self.dataloader:
                inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                diff = self.compare_batch(model1, model2, inputs, outputs)    
            
                total_diff += diff.item()

        if self.reduction == 'mean':
            total_diff /= len(self.dataloader)

        return total_diff


class LPOutputs(OutputComparison):
    """Calculate the L-p distance between two models' outputs."""
    type_ = 'metric'

    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu', p=2, eps=1e-06):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)
        self.p = p
        self.eps = eps
        self.criterion = nn.PairwiseDistance(p=p, eps=1e-06, keepdim=False)

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        return self.criterion(outputs1, outputs2)


class L1Outputs(LPOutputs):
    """Calculate the L1 divergence between two models."""
    type_ = 'metric'

    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device, p=1)


class L2Outputs(LPOutputs):
    """Calculate the L2 divergence between two models."""
    type_ = 'metric'

    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device, p=2)


class CrossEntropy(OutputComparison):
    """Calculate the cross-entropy between two models."""
    type_ = 'divergence'

    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        return self.criterion(outputs1, outputs2)


class KLDivergence(OutputComparison):
    """Calculate the KL divergence between two models."""
    type_ = 'divergence'

    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)
        self.criterion = nn.KLDivLoss

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        log_softmax1 = torch.nn.functional.log_softmax(outputs1, dim=1)
        softmax2 = torch.nn.functional.softmax(outputs2, dim=1)

        return self.criterion(log_softmax1, softmax2)


class SupNormOutputs(OutputComparison):
    """Calculate the sup-norm between the outputs of model1 and model2."""
    type_ = 'metric'

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



