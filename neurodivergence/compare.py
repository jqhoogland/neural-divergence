from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class Comparison:
    """The base class for all neural network comparisons."""

    def __init__(self, reduction='mean', device='cpu', *args, **kwargs):
        self.reduction = reduction
        self.device = device

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


class DataDependentComparisonMixin:
    """
    A mixin for data-dependent comparisons. 
    Yes, mixins are often considered harmful, but GPT-4 called this a reasonable idea.
    """

    def __init__(self, dataloader: Optional[DataLoader]=None, *args, **kwargs):
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
        total_samples = 0

        with self.eval(model1, model2):
            for inputs, outputs in self.dataloader:
                inputs, outputs = inputs.to(self.device), outputs.to(self.device)
                diff = self.compare_batch(model1, model2, inputs, outputs)    
            
                total_diff += diff.item()
                total_samples += inputs.size(0)

        return total_diff / total_samples

class Metric(Comparison):
    """The base class for all neural network metrics."""
    

class Divergence(Metric):
    """The base class for all neural network divergences."""

    def compare_symmetrized(self, model1: nn.Module, model2: nn.Module):
        """Compare model1 and model2 symmetrically."""
        return (self(model1, model2) + self(model2, model1)) / 2


class KLDivergence(Divergence, DataDependentComparisonMixin):
    """Calculate the KL divergence between two models."""
    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)
        self.criterion = nn.KLDivLoss(reduction='sum')

        if reduction != 'mean':       
            raise NotImplementedError

    def compare_batch(self, model1: nn.Module, model2: nn.Module, inputs: Tensor, outputs: Tensor):
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        log_softmax1 = torch.nn.functional.log_softmax(outputs1, dim=1)
        softmax2 = torch.nn.functional.softmax(outputs2, dim=1)

        return self.criterion(log_softmax1, softmax2)


class SupNorm(Divergence, DataDependentComparisonMixin):
    """Calculate the sup-norm between the outputs of model1 and model2."""

    def __init__(self, dataloader: DataLoader, reduction='mean', device='cpu'):
        super().__init__(dataloader=dataloader, reduction=reduction, device=device)

        if reduction != 'mean':       
            raise NotImplementedError

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


