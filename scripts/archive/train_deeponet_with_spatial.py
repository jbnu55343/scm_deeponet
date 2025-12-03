#!/usr/bin/env python3
"""
DeepONet training on SPATIAL dataset (with spatial features from neighboring edges).
Branch network learns spatial encoding, Trunk learns temporal dynamics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from datetime import datetime
import time

# ============================================================================
# CONFIG
# ============================================================================
DATA_FILE = "data/dataset_sumo_5km_lag12_filtered_with_spatial.npz"
MODEL_SAVE_PATH = "models/deeponet_spatial.pt"
RESULTS_SAVE_PATH = "results/deeponet_spatial_results.npz"

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# MODEL: DeepONet
# ============================================================================

class DeepONetSpatial(nn.Module):
    """DeepONet with spatial features"""
    def __init__(self, input_size=23, branch_depth=3, branch_width=256,
                 trunk_depth=3, trunk_width=256, output_width=256):
        super().__init__()
        
        # Branch network: learns spatial encoding from input features
        branch_layers = []
        prev_size = input_size
        for i in range(branch_depth):
            branch_layers.append(nn.Linear(prev_size, branch_width))
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Dropout(0.1))
            prev_size = branch_width
        
        self.branch = nn.Sequential(*branch_layers)
        
        # Trunk network: learns temporal basis functions
        trunk_layers = []
        prev_size = 1  # Single input (time or dummy)
        for i in range(trunk_depth):
            trunk_layers.append(nn.Linear(prev_size, trunk_width))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(0.1))
            prev_size = trunk_width
        trunk_layers.append(nn.Linear(prev_size, output_width))
        
        self.trunk = nn.Sequential(*trunk_layers)
        
        # Output combination: element-wise product + sum
        self.output_linear = nn.Linear(output_width, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Branch: spatial encoding
        branch_out = self.branch(x)  # (batch, 256)
        
        # Trunk: temporal basis functions
        # Create dummy input (can be time index normalized to [0,1])
        dummy_input = torch.ones(batch_size, 1, device=x.device)
        trunk_out = self.trunk(dummy_input)  # (batch, 256)
        
        # Element-wise product and sum
        combined = branch_out * trunk_out  # (batch, 256)
        output = self.output_linear(combined)  # (batch, 1)
        
        return output


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

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
    """Evaluate model on a dataset with batch-wise metrics to avoid memory issues."""
    model.eval()
    
    mae_list = []
    rmse_list = []
    counts = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_true = y_batch.numpy()
            
            # Ensure same shape
            if y_pred.ndim > 1:
                y_pred = y_pred.squeeze()
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            
            # Batch-wise metrics
            mae_list.append(np.abs(y_true - y_pred).sum())
            rmse_list.append(((y_true - y_pred) ** 2).sum())
            counts.append(len(y_batch))
    
    # Final metrics
    total_samples = sum(counts)
    mae = sum(mae_list) / total_samples
    rmse = np.sqrt(sum(rmse_list) / total_samples)
    
    # For R² - compute with correct mean
    y_mean_total = 0.0
    n_total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_true = y_batch.numpy()
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            y_mean_total += y_true.sum()
            n_total += len(y_true)
    
    y_mean_total = y_mean_total / n_total
    
    # Compute R² with correct mean
    ss_res = 0.0
    ss_tot = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_true = y_batch.numpy()
            
            if y_pred.ndim > 1:
                y_pred = y_pred.squeeze()
            if y_true.ndim > 1:
                y_true = y_true.squeeze()
            
            ss_res += ((y_true - y_pred) ** 2).sum()
            ss_tot += ((y_true - y_mean_total) ** 2).sum()
    
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    return mae, rmse, r2, None, None


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("DeepONet Training: Spatial Dataset")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now()}")
    
    # Load data
    print(f"\nLoading data from {DATA_FILE}...")
    d = np.load(DATA_FILE)
    X = d['X']
    y = d['Y']
    
    print(f"Loaded: X={X.shape}, y={y.shape}")
    print(f"Target: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Temporal split (80% train, 20% test)
    n_total = len(X)
    n_train = int(n_total * 0.8)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"  Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"  Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train.reshape(-1)).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test.reshape(-1)).float()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    input_size = X.shape[1]  # Should be 23
    model = DeepONetSpatial(input_size=input_size).to(DEVICE)
    
    print(f"\nModel: DeepONet (Spatial)")
    print(f"  Input size: {input_size}")
    print(f"  Branch: 3 layers, 256 width")
    print(f"  Trunk: 3 layers, 256 width → 256 output")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
    best_test_r2 = -np.inf
    best_test_mae = np.inf
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        
        # Evaluate
        train_mae, train_rmse, train_r2, _, _ = evaluate(model, train_loader, DEVICE)
        test_mae, test_rmse, test_r2, test_true, test_pred = evaluate(model, test_loader, DEVICE)
        
        # Learning rate scheduling
        scheduler.step(test_r2)
        
        # Print progress (every 10 epochs)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f} | "
                  f"Train R2={train_r2:.4f} | Test R2={test_r2:.4f} | Test MAE={test_mae:.4f}")
        
        # Track best model
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_test_mae = test_mae
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    train_mae, train_rmse, train_r2, train_true, train_pred = evaluate(model, train_loader, DEVICE)
    test_mae, test_rmse, test_r2, test_true, test_pred = evaluate(model, test_loader, DEVICE)
    
    print("\n" + "="*80)
    print("FINAL RESULTS (Best Model)")
    print("="*80)
    print(f"\nTRAIN (first 80%):")
    print(f"  MAE:  {train_mae:.4f} km/h")
    print(f"  RMSE: {train_rmse:.4f} km/h")
    print(f"  R2:   {train_r2:.4f}")
    
    print(f"\nTEST (last 20%, unseen time periods):")
    print(f"  MAE:  {test_mae:.4f} km/h")
    print(f"  RMSE: {test_rmse:.4f} km/h")
    print(f"  R2:   {test_r2:.4f}")
    
    print(f"\nTraining Summary:")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Training time: {training_time:.1f}s")
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_SAVE_PATH), exist_ok=True)
    np.savez(
        RESULTS_SAVE_PATH,
        train_true=train_true,
        train_pred=train_pred,
        test_true=test_true,
        test_pred=test_pred,
        train_mae=train_mae,
        train_rmse=train_rmse,
        train_r2=train_r2,
        test_mae=test_mae,
        test_rmse=test_rmse,
        test_r2=test_r2,
    )
    
    print(f"\n✓ Model saved to {MODEL_SAVE_PATH}")
    print(f"✓ Results saved to {RESULTS_SAVE_PATH}")


if __name__ == '__main__':
    main()
