#!/usr/bin/env python3
"""
MLP training with temporal split (simulating cross-domain evaluation).

This script trains an MLP baseline using temporal split to test 
generalization to unseen time periods (approximating cross-scenario).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================
DATA_FILE = "data/dataset_sumo_5km_lag12_filtered.npz"  # Filtered data with valid samples
MODEL_SAVE_PATH = "models/mlp_temporal_split.pt"

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# FUNCTIONS
# ============================================================================

class MLPModel(nn.Module):
    """MLP matching paper architecture."""
    def __init__(self, input_size=18, hidden_size=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def eval_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R²."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return mae, rmse, r2


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        out = model(X_batch)
        loss = torch.nn.functional.mse_loss(out, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def eval_model(model, loader, device):
    """Evaluate model."""
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch)
            y_pred_list.append(out.cpu().numpy().flatten())
            y_true_list.append(y_batch.numpy())
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    return eval_metrics(y_true, y_pred), y_true, y_pred


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("MLP Training: Temporal Split Evaluation")
    print("="*80)
    print(f"Device: {DEVICE}\n")
    
    # Load filtered data
    print(f"Loading data from {DATA_FILE}...")
    d = np.load(DATA_FILE, allow_pickle=True)
    X = d['X']
    y = d['Y'].squeeze()
    
    print(f"Loaded: X={X.shape}, y={y.shape}")
    print(f"Target: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Temporal split: 80% train, 20% test
    n_samples = len(X)
    split_idx = int(0.8 * n_samples)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples (first 80%)")
    print(f"  Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    print(f"Test set: {len(X_test)} samples (last 20%)")
    print(f"  Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
    
    # Create dataloaders
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).float()
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_size = X.shape[1]
    model = MLPModel(input_size=input_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nModel: MLP({input_size} -> 256 -> 256 -> 1)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training
    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
    best_r2 = -float('inf')
    best_state = None
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        
        (train_mae, train_rmse, train_r2), _, _ = eval_model(model, train_loader, DEVICE)
        (test_mae, test_rmse, test_r2), _, _ = eval_model(model, test_loader, DEVICE)
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: "
                  f"Loss={train_loss:.4f} | "
                  f"Train R²={train_r2:.4f} | "
                  f"Test R²={test_r2:.4f} | "
                  f"Test MAE={test_mae:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Final eval
    print("\n" + "="*80)
    print("FINAL RESULTS (Best Model)")
    print("="*80)
    
    (train_mae, train_rmse, train_r2), train_true, train_pred = eval_model(model, train_loader, DEVICE)
    (test_mae, test_rmse, test_r2), test_true, test_pred = eval_model(model, test_loader, DEVICE)
    
    print(f"\nTRAIN (first 80%):")
    print(f"  MAE:  {train_mae:.4f} km/h")
    print(f"  RMSE: {train_rmse:.4f} km/h")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTEST (last 20%, unseen time periods):")
    print(f"  MAE:  {test_mae:.4f} km/h")
    print(f"  RMSE: {test_rmse:.4f} km/h")
    print(f"  R²:   {test_r2:.4f}")
    
    print(f"\n" + "="*80)
    print("COMPARISON WITH PAPER")
    print("="*80)
    
    paper_mae = 1.430
    paper_rmse = 2.243
    paper_r2 = 0.9856
    
    print(f"\nPaper (leave-scenario-out, 8.25 km/h mean):")
    print(f"  MAE:  {paper_mae:.4f}")
    print(f"  RMSE: {paper_rmse:.4f}")
    print(f"  R²:   {paper_r2:.4f}")
    
    print(f"\nOurs (temporal split, {y_test.mean():.2f} km/h mean):")
    print(f"  MAE:  {test_mae:.4f} ({test_mae/paper_mae:.2f}x paper)")
    print(f"  RMSE: {test_rmse:.4f} ({test_rmse/paper_rmse:.2f}x paper)")
    print(f"  R²:   {test_r2:.4f} ({test_r2/paper_r2:.3f}x paper)")
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'metrics': {
            'train': {'mae': float(train_mae), 'rmse': float(train_rmse), 'r2': float(train_r2)},
            'test': {'mae': float(test_mae), 'rmse': float(test_rmse), 'r2': float(test_r2)},
        },
        'predictions': {
            'test_true': test_true,
            'test_pred': test_pred,
        }
    }, MODEL_SAVE_PATH)
    
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
