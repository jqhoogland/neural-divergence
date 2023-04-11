import torch
import torch.nn as nn

class FCNN(nn.Module):
    
    def __init__(self, bias=True, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(28 * 28, 128, bias=bias)
        self.fc2 = nn.Linear(128, 64, bias=bias)
        self.fc3 = nn.Linear(64, 10, bias=bias)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
