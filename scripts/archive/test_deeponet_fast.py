#!/usr/bin/env python3
"""
Fast DeepONet training test with small subset for comparison with MLP.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

# ============================================================================
# DEEPONET ARCHITECTURE
# ============================================================================

class DeepONet(nn.Module):
    """DeepONet: Branch net + Trunk net"""
    def __init__(self, input_size=19, branch_depth=2, trunk_depth=2, width=128):
        super().__init__()
        
        # Branch net: processes input features
        branch_layers = []
        prev_size = input_size
        for _ in range(branch_depth):
            branch_layers.extend([
                nn.Linear(prev_size, width),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = width
        self.branch = nn.Sequential(*branch_layers, nn.Linear(prev_size, width))
        
        # Trunk net: fixed neural operator
        trunk_layers = []
        for _ in range(trunk_depth):
            trunk_layers.extend([
                nn.Linear(1, width),  # Dummy input (will be used for flexibility)
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.trunk = nn.Sequential(*trunk_layers, nn.Linear(width, width))
    
    def forward(self, x):
        # Branch output
        branch_out = self.branch(x)  # (batch, width)
        
        # Trunk output (use constant input for simplicity)
        dummy_trunk_input = torch.ones(x.shape[0], 1, device=x.device)
        trunk_out = self.trunk(dummy_trunk_input)  # (batch, width)
        
        # Combine: element-wise product then sum
        combined = branch_out * trunk_out  # (batch, width)
        output = combined.sum(dim=1, keepdim=True)  # (batch, 1)
        
        return output


# ============================================================================
# TRAINING
# ============================================================================

# Load data
print("Loading filtered data...")
d = np.load("data/dataset_sumo_5km_lag12_filtered.npz", allow_pickle=True)
X = d['X']
y = d['Y'].squeeze()

# Use only 10% for quick test
subset_size = len(X) // 10
X = X[:subset_size]
y = y[:subset_size]

print(f"Using subset: {len(X)} samples")
print(f"Target: mean={y.mean():.4f}, std={y.std():.4f}")

# Split
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Create loaders
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).float()
X_test_t = torch.from_numpy(X_test).float()
y_test_t = torch.from_numpy(y_test).float()

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=512, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=512)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepONet(input_size=X.shape[1], branch_depth=2, trunk_depth=2, width=128).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(f"\nDeepONet Training on {device}...")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

for epoch in range(10):
    start = time.time()
    
    # Train
    model.train()
    train_loss = 0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device).unsqueeze(1)
        optimizer.zero_grad()
        out = model(X_b)
        loss = torch.nn.functional.mse_loss(out, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Eval
    model.eval()
    with torch.no_grad():
        train_pred = []
        test_pred = []
        for X_b, _ in train_loader:
            train_pred.append(model(X_b.to(device)).cpu())
        for X_b, _ in test_loader:
            test_pred.append(model(X_b.to(device)).cpu())
        train_pred = torch.cat(train_pred).numpy().flatten()
        test_pred = torch.cat(test_pred).numpy().flatten()
        
        train_r2 = 1 - np.sum((y_train - train_pred) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
        test_r2 = 1 - np.sum((y_test - test_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    
    elapsed = time.time() - start
    print(f"Epoch {epoch+1:2d}: Loss={train_loss:.4f} | "
          f"Train R²={train_r2:.4f} | Test R²={test_r2:.4f} | "
          f"Time={elapsed:.1f}s")

print("\n✓ DeepONet test completed!")
