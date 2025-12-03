import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.fc = nn.Linear(hidden_dim * input_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim

    def forward(self, x):
        # x: (B, F) -> (B, 1, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        print(f"TCN Input shape after unsqueeze: {x.shape}")
        x = F.relu(self.conv1(x))
        print(f"After conv1: {x.shape}")
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        print(f"After conv2: {x.shape}")
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        print(f"After view: {x.shape}")
        x = self.fc(x)
        return x

def test():
    model = TCN(input_dim=23)
    x = torch.randn(32, 23)
    try:
        out = model(x)
        print("Forward pass successful")
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")

if __name__ == "__main__":
    test()
