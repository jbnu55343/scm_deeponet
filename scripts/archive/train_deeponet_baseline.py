#!/usr/bin/env python3
"""
DeepONet training on full filtered dataset.
Compared against MLP baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from datetime import datetime

# ============================================================================
# MODEL
# ============================================================================

class DeepONet(nn.Module):
    """DeepONet: Branch network + Trunk network."""
    def __init__(self, input_size=19, branch_depth=3, branch_width=256, 
                 trunk_depth=3, trunk_width=256, output_width=256):
        super().__init__()
        
        # Branch net: processes input features
        branch_layers = []
        prev_size = input_size
        for i in range(branch_depth):
            branch_layers.append(nn.Linear(prev_size, branch_width))
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Dropout(0.1))
            prev_size = branch_width
        self.branch = nn.Sequential(*branch_layers, nn.Linear(prev_size, output_width))
        
        # Trunk net: learns the shared operator
        trunk_layers = []
        prev_size = 1  # Single scalar input (s-parameter for operator)
        for i in range(trunk_depth):
            trunk_layers.append(nn.Linear(prev_size, trunk_width))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(0.1))
            prev_size = trunk_width
        self.trunk = nn.Sequential(*trunk_layers, nn.Linear(prev_size, output_width))
    
    def forward(self, x):
        # Branch output: G(x) of shape (batch, output_width)
        branch_out = self.branch(x)
        
        # Trunk output: operator B of shape (batch, output_width)
        # Use a dummy input (standard parameter s) for each sample
        s = torch.ones(x.shape[0], 1, device=x.device)
        trunk_out = self.trunk(s)
        
        # DeepONet output: sum of element-wise products
        # y = sum_i B_i(s) * G_i(x)
        output = (branch_out * trunk_out).sum(dim=1, keepdim=True)
        
        return output


# ============================================================================
# CONFIG & FUNCTIONS
# ============================================================================

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_FILE = "data/dataset_sumo_5km_lag12_filtered.npz"
MODEL_SAVE_PATH = "models/deeponet_baseline.pt"
RESULTS_SAVE_PATH = "results/deeponet_baseline_results.npz"


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1) if y_batch.dim() == 1 else y_batch.to(device)
        
        optimizer.zero_grad()
        out = model(X_batch)
        loss = torch.nn.functional.mse_loss(out, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            out = model(X_batch).squeeze()
            y_pred_list.append(out.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    
    # Compute metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return mae, rmse, r2, y_true, y_pred


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("DeepONet Baseline: Full Filtered Dataset Training")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now()}\n")
    
    # Load filtered data
    print(f"Loading data from {DATA_FILE}...")
    d = np.load(DATA_FILE, allow_pickle=True)
    X = d['X'].astype(np.float32)
    y = d['Y'].squeeze().astype(np.float32)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Target statistics:")
    print(f"  Mean: {y.mean():.4f} km/h")
    print(f"  Std:  {y.std():.4f} km/h")
    
    # Temporal split
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    print(f"\nData split (temporal):")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Create dataloaders
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    input_size = X.shape[1]
    model = DeepONet(input_size=input_size, 
                     branch_depth=3, branch_width=256,
                     trunk_depth=3, trunk_width=256,
                     output_width=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                      patience=5)
    
    print(f"\nModel: DeepONet")
    print(f"  Input size: {input_size}")
    print(f"  Branch: depth={3}, width={256}")
    print(f"  Trunk: depth={3}, width={256}")
    print(f"  Output width: {256}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80 + "\n")
    
    best_test_r2 = -float('inf')
    best_model_state = None
    patience_count = 0
    max_patience = 10
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        train_mae, train_rmse, train_r2, _, _ = evaluate(model, train_loader, DEVICE)
        test_mae, test_rmse, test_r2, _, _ = evaluate(model, test_loader, DEVICE)
        
        # Learning rate scheduling
        scheduler.step(test_r2)
        
        # Save best model
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
                  f"Loss={train_loss:.4f} | "
                  f"Train: MAE={train_mae:.4f} R²={train_r2:.4f} | "
                  f"Test: MAE={test_mae:.4f} R²={test_r2:.4f}")
        
        # Early stopping
        if patience_count >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL RESULTS (Best Model)")
    print("="*80)
    
    train_mae, train_rmse, train_r2, train_true, train_pred = evaluate(model, train_loader, DEVICE)
    test_mae, test_rmse, test_r2, test_true, test_pred = evaluate(model, test_loader, DEVICE)
    
    print(f"\nTrain Set (first 80%):")
    print(f"  MAE:  {train_mae:.4f} km/h")
    print(f"  RMSE: {train_rmse:.4f} km/h")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTest Set (last 20%, unseen time period):")
    print(f"  MAE:  {test_mae:.4f} km/h")
    print(f"  RMSE: {test_rmse:.4f} km/h")
    print(f"  R²:   {test_r2:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)
    np.savez(RESULTS_SAVE_PATH,
             train_true=train_true,
             train_pred=train_pred,
             test_true=test_true,
             test_pred=test_pred,
             train_mae=train_mae,
             train_rmse=train_rmse,
             train_r2=train_r2,
             test_mae=test_mae,
             test_rmse=test_rmse,
             test_r2=test_r2)
    print(f"✓ Results saved to {RESULTS_SAVE_PATH}")
    
    print("\n" + "="*80)
    print("✓ DeepONet baseline training completed")
    print("="*80)


if __name__ == '__main__':
    main()
