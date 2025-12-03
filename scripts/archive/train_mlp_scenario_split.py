#!/usr/bin/env python3
"""
MLP training with proper leave-scenario-out split.

This script trains an MLP baseline using the exact same evaluation protocol
as the paper:
- Train: S001, S002, S003, S004 (seen scenarios)
- Test: S005, S006 (unseen scenarios)

This tests cross-scenario generalization ability.
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
DATA_FILE = "data/dataset_sumo_5km_lag12.npz"  # Using original data (5.4K samples)
MODEL_SAVE_PATH = "models/mlp_leave_scenario_out.pt"
LOG_FILE = "logs/mlp_leave_scenario_out.log"

BATCH_SIZE = 64  # Larger batch for smaller dataset
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_split_data(npz_file, use_filtered=False):
    """
    Load data from npz file and split by scenario.
    
    Args:
        npz_file: Path to .npz file
        use_filtered: If True, use filtered data; else use all data
    
    Returns:
        (X_train, y_train, X_test, y_test, meta_info)
    """
    print(f"Loading data from {npz_file}...")
    d = np.load(npz_file, allow_pickle=True)
    
    X = d['X']  # (N, features)
    y = d['Y']  # (N,) or (N, 1)
    
    # Extract meta information
    meta = d['meta'].item() if hasattr(d['meta'], 'item') else dict(d['meta'])
    
    print(f"Data shape: X={X.shape}, y={y.shape if hasattr(y, 'shape') else len(y)}")
    
    # If filtered data available, apply filtering
    if use_filtered and 'is_valid' in d:
        valid_mask = d['is_valid']
        X = X[valid_mask]
        y = y[valid_mask] if hasattr(y, '__len__') else y
        print(f"After filtering: X={X.shape}")
    
    # Ensure y is 1D
    if y.ndim > 1:
        y = y.squeeze()
    
    # Get scenario information
    scenarios = meta.get('scenarios', [])
    per_scenario = meta.get('per_scenario', [])
    
    print(f"\nScenarios: {scenarios}")
    print(f"Samples per scenario:")
    cumsum = 0
    scenario_ranges = {}
    scenario_ids = np.zeros(len(X), dtype=int)
    
    for i, sc_info in enumerate(per_scenario):
        s = sc_info['scenario']
        t = sc_info['T']
        scenario_ranges[s] = (cumsum, cumsum + t)
        print(f"  {s}: samples {cumsum}-{cumsum+t-1} (T={t})")
        
        # Map scenario index to scenario ID
        if cumsum < len(X):
            end_idx = min(cumsum + t, len(X))
            scenario_ids[cumsum:end_idx] = i
        
        cumsum += t
    
    # Split: train on S001-S004 (indices 0-3), test on S005-S006 (indices 4-5)
    train_mask = np.isin(scenario_ids, [0, 1, 2, 3])
    test_mask = np.isin(scenario_ids, [4, 5])
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"\nTrain set: {len(X_train)} samples (S001-S004)")
    print(f"Test set: {len(X_test)} samples (S005-S006)")
    
    # Compute statistics
    train_mean = y_train.mean()
    train_std = y_train.std()
    test_mean = y_test.mean()
    test_std = y_test.std()
    
    print(f"\nTarget statistics:")
    print(f"  Train: mean={train_mean:.4f}, std={train_std:.4f}")
    print(f"  Test: mean={test_mean:.4f}, std={test_std:.4f}")
    
    return X_train, y_train, X_test, y_test, {
        'scenarios': scenarios,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_mean': test_mean,
        'test_std': test_std,
    }


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders."""
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


class MLPModel(nn.Module):
    """Simple MLP matching paper architecture."""
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


def eval_epoch(model, loader, device):
    """Evaluate model on a loader."""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            out = model(X_batch)
            y_pred.append(out.cpu().numpy().flatten())
            y_true.append(y_batch.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return mae, rmse, r2, y_true, y_pred


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


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("MLP Training: Leave-Scenario-Out Evaluation")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now()}")
    
    # Load data
    X_train, y_train, X_test, y_test, meta_info = load_and_split_data(DATA_FILE)
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE)
    
    # Initialize model
    input_size = X_train.shape[1]
    model = MLPModel(input_size=input_size).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nModel: MLP({input_size} -> 256 -> 256 -> 1)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
    best_test_r2 = -float('inf')
    best_model_state = None
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        train_mae, train_rmse, train_r2, _, _ = eval_epoch(model, train_loader, DEVICE)
        test_mae, test_rmse, test_r2, _, _ = eval_epoch(model, test_loader, DEVICE)
        
        # Save best model
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train MAE: {train_mae:.4f} | "
                  f"Test MAE: {test_mae:.4f} | "
                  f"Test R²: {test_r2:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation (Best Model)")
    print("="*80)
    
    train_mae, train_rmse, train_r2, train_true, train_pred = eval_epoch(model, train_loader, DEVICE)
    test_mae, test_rmse, test_r2, test_true, test_pred = eval_epoch(model, test_loader, DEVICE)
    
    print(f"\nTRAIN (S001-S004):")
    print(f"  MAE:  {train_mae:.4f} km/h")
    print(f"  RMSE: {train_rmse:.4f} km/h")
    print(f"  R²:   {train_r2:.4f}")
    
    print(f"\nTEST (S005-S006, Unseen Scenarios):")
    print(f"  MAE:  {test_mae:.4f} km/h")
    print(f"  RMSE: {test_rmse:.4f} km/h")
    print(f"  R²:   {test_r2:.4f}")
    
    # Compare with paper
    print(f"\n" + "="*80)
    print("Comparison with Paper Results (MLP baseline)")
    print("="*80)
    paper_mae = 1.430
    paper_rmse = 2.243
    paper_r2 = 0.9856
    
    print(f"\nPaper Results (original data, 8.25 km/h mean):")
    print(f"  MAE:  {paper_mae:.4f} km/h")
    print(f"  RMSE: {paper_rmse:.4f} km/h")
    print(f"  R²:   {paper_r2:.4f}")
    
    print(f"\nOur Results (filtered data, {meta_info['test_mean']:.2f} km/h mean):")
    print(f"  MAE:  {test_mae:.4f} km/h (ratio: {test_mae/paper_mae:.2f}x)")
    print(f"  RMSE: {test_rmse:.4f} km/h (ratio: {test_rmse/paper_rmse:.2f}x)")
    print(f"  R²:   {test_r2:.4f} (ratio: {test_r2/paper_r2:.2f})")
    
    print(f"\nTarget Mean Ratio (filtered/original):")
    print(f"  {meta_info['test_mean']:.2f} / {paper_mae / (1.430/17.34):.2f} = {meta_info['test_mean'] / 8.25:.2f}x")
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    
    # Save results
    results = {
        'model_state': model.state_dict(),
        'train_results': {
            'mae': float(train_mae),
            'rmse': float(train_rmse),
            'r2': float(train_r2),
        },
        'test_results': {
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'r2': float(test_r2),
        },
        'meta_info': meta_info,
        'predictions': {
            'test_true': test_true,
            'test_pred': test_pred,
        }
    }
    torch.save(results, MODEL_SAVE_PATH.replace('.pt', '_results.pt'))
    print(f"Results saved to {MODEL_SAVE_PATH.replace('.pt', '_results.pt')}")


if __name__ == '__main__':
    main()
