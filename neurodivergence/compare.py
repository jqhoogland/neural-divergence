import torch
import torch.nn as nn

# functional comparions

def kl_divergence_between_models(model1: nn.Module, model2: nn.Module, data_loader, device='cpu'):
    """Calculate the KL divergence between model1 and model2."""
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    criterion = nn.KLDivLoss(reduction='batchmean')
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs1 = model1(data)
            outputs2 = model2(data)

            log_softmax1 = torch.log_softmax(outputs1, dim=1)
            softmax2 = torch.softmax(outputs2, dim=1)

            # Calculate the kl-divergence
            kl_loss = criterion(log_softmax1, softmax2)

            total_loss += kl_loss.item() * data.size(0)
            total_samples += data.size(0)

    return total_loss / total_samples


def sup_norm_outputs_between_models(model1: nn.Module, model2: nn.Module, data_loader, device='cpu'):
    """Calculate the sup-norm between the outputs of model1 and model2."""
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()

    sup_norm = 0.0

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            outputs1 = model1(data)
            softmax1 = torch.softmax(outputs1, dim=1)
            
            outputs2 = model2(data)
            softmax2 = torch.softmax(outputs2, dim=1)

            # Calculate the sup norm for the current batch
            batch_sup_norm = torch.max(torch.abs(softmax1 - softmax2)).item()

            # Update the overall sup norm
            sup_norm = max(sup_norm, batch_sup_norm)

    return sup_norm


