#!/usr/bin/env python3
"""
DeepONet Training on METR-LA (or any spatial-temporal dataset)
Branch: Spatial features (Neighbors)
Trunk: Temporal features (History)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
from sklearn.metrics import mean_absolute_error, r2_score

class DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, hidden_dim=128, output_dim=128):
        super().__init__()
        
        # Branch Net: Process Spatial Context
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Trunk Net: Process Temporal History
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_branch, x_trunk):
        # x_branch: (B, branch_dim)
        # x_trunk: (B, trunk_dim)
        
        b_out = self.branch(x_branch) # (B, P)
        t_out = self.trunk(x_trunk)   # (B, P)
        
        # Dot product
        out = torch.sum(b_out * t_out, dim=1, keepdim=True) + self.bias
        return out

class DualInputDataset(Dataset):
    def __init__(self, X_trunk, X_branch, Y, stats=None):
        self.X_trunk = torch.FloatTensor(X_trunk)
        self.X_branch = torch.FloatTensor(X_branch)
        self.Y = torch.FloatTensor(Y).unsqueeze(1)
        self.stats = stats
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        xt = self.X_trunk[idx]
        xb = self.X_branch[idx]
        y = self.Y[idx]
        
        if self.stats:
            xt = (xt - self.stats['xt_mean']) / (self.stats['xt_std'] + 1e-8)
            xb = (xb - self.stats['xb_mean']) / (self.stats['xb_std'] + 1e-8)
            y = (y - self.stats['y_mean']) / (self.stats['y_std'] + 1e-8)
            
        return xt, xb, y

def load_data(npz_path):
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    
    # X is Temporal (Trunk), X_spatial is Spatial (Branch)
    X_trunk = d['X']
    if 'X_spatial' in d:
        X_branch = d['X_spatial']
    else:
        # Fallback if no spatial features (should not happen for DeepONet)
        print("Warning: No X_spatial found, using zeros")
        X_branch = np.zeros((X_trunk.shape[0], 1))

    Y = d['Y']
    
    # Split
    split = d['split'].item()
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    return (X_trunk, X_branch, Y), (train_idx, val_idx, test_idx)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    (X_trunk, X_branch, Y), (train_idx, val_idx, test_idx) = load_data(args.npz)
    
    # Calculate Stats from Training Data
    print("Calculating statistics from training set...")
    xt_train = X_trunk[train_idx]
    xb_train = X_branch[train_idx]
    y_train = Y[train_idx]
    
    # Helper to safely convert to tensor
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return torch.tensor(x).float()

    stats = {
        'xt_mean': to_tensor(xt_train.mean(0)),
        'xt_std': to_tensor(xt_train.std(0)),
        'xb_mean': to_tensor(xb_train.mean(0)),
        'xb_std': to_tensor(xb_train.std(0)),
        'y_mean': to_tensor(y_train.mean(0)),
        'y_std': to_tensor(y_train.std(0))
    }
    
    print(f"Trunk dim: {X_trunk.shape[1]}")
    print(f"Branch dim: {X_branch.shape[1]}")
    print(f"Y Mean: {stats['y_mean'].item():.2f}, Y Std: {stats['y_std'].item():.2f}")
    
    # Datasets
    train_ds = DualInputDataset(X_trunk[train_idx], X_branch[train_idx], Y[train_idx], stats)
    val_ds = DualInputDataset(X_trunk[val_idx], X_branch[val_idx], Y[val_idx], stats)
    test_ds = DualInputDataset(X_trunk[test_idx], X_branch[test_idx], Y[test_idx], stats)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    # Model
    model = DeepONet(
        branch_dim=X_branch.shape[1],
        trunk_dim=X_trunk.shape[1],
        hidden_dim=128,
        output_dim=128
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    # Move stats to device for denormalization
    y_mean_dev = stats['y_mean'].to(device)
    y_std_dev = stats['y_std'].to(device)
    
    print("\nStarting training...")
    start_train_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for xt, xb, y in train_loader:
            xt, xb, y = xt.to(device), xb.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(xb, xt)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0
        preds = []
        trues = []
        with torch.no_grad():
            for xt, xb, y in val_loader:
                xt, xb, y = xt.to(device), xb.to(device), y.to(device)
                pred = model(xb, xt)
                loss = criterion(pred, y)
                val_loss += loss.item() * y.size(0)
                
                # Denormalize for metrics
                pred_real = pred * y_std_dev + y_mean_dev
                y_real = y * y_std_dev + y_mean_dev
                
                preds.append(pred_real.cpu().numpy())
                trues.append(y_real.cpu().numpy())
        
        val_loss /= len(val_ds)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        
        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save)
            print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.2f} | R2: {r2:.4f} *")
        elif (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.2f} | R2: {r2:.4f}")
            
    training_time_total = time.time() - start_train_time

    # Final Test
    model.load_state_dict(torch.load(args.save))
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xt, xb, y in test_loader:
            xt, xb, y = xt.to(device), xb.to(device), y.to(device)
            pred = model(xb, xt)
            
            pred_real = pred * y_std_dev + y_mean_dev
            y_real = y * y_std_dev + y_mean_dev
            
            preds.append(pred_real.cpu().numpy())
            trues.append(y_real.cpu().numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    test_mae = mean_absolute_error(trues, preds)
    test_rmse = np.sqrt(((preds - trues)**2).mean())
    test_r2 = r2_score(trues, preds)
    
    print(f"\n[TEST] MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f} | R2: {test_r2:.4f}")
    
    # Standardized Output
    import json
    num_params = sum(p.numel() for p in model.parameters())
    # Note: DeepONet training time wasn't tracked in the loop, adding rough estimate or need to wrap loop
    # Let's wrap the loop in next edit or just use current time if loop finished
    # Actually, I should have added start_time before loop.
    # Since I can't easily edit the loop start without reading more context, I'll skip time for now or add it if I edit the loop.
    # Wait, I can edit the loop start too.
    
    result = {
        "model": "DeepONet",
        "params": num_params,
        "time_sec": training_time_total,
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2)
    }
    print("FINAL_RESULT_JSON:" + json.dumps(result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', default='models/deeponet_metr_la.pth')
    args = parser.parse_args()
    
    train(args)
