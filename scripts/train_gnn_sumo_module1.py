import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

class SumoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class GNNLocalModule1(nn.Module):
    def __init__(self, center_dim=19, hidden_dim=64, dropout=0.1):
        super().__init__()
        # Module 1 has no spatial neighbors, so we only use the center node part.
        # This effectively makes it an MLP, but we keep the architecture similar to GNNLocal's center part.
        
        self.fc_center = nn.Linear(center_dim, hidden_dim)
        # self.fc_neighbor = nn.Linear(neighbor_dim, hidden_dim) # Removed
        
        # Graph Convolution (Self-loop only)
        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        # self.w_neighbor = nn.Linear(hidden_dim, hidden_dim) # Removed
        
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, 19)
        
        # 1. Message Passing (Only Self)
        h_center = F.relu(self.fc_center(x))
        h_center = self.dropout(h_center)
        
        # Aggregate: Only self
        h_agg = self.w_self(h_center)
        h_agg = F.relu(h_agg)
        
        # 2. Readout
        out = self.fc_out(h_agg)
        return out.squeeze(-1)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, default='data/dataset_sumo_5km_lag12_filtered.npz')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading {args.npz}...")
    data = np.load(args.npz, allow_pickle=True)
    
    # Handle keys
    if 'split' in data:
        X = data['X']
        Y = data['Y']
        split = data['split'].item()
        train_idx = split['train']
        test_idx = split['test']
        X_train = X[train_idx]
        y_train = Y[train_idx]
        X_test = X[test_idx]
        y_test = Y[test_idx]
    else:
        # Try external split
        split_path = os.path.join(os.path.dirname(args.npz), 'sumo_split_standard.json')
        if os.path.exists(split_path):
            print(f"Loading split from {split_path}...")
            with open(split_path, 'r') as f:
                split = json.load(f)
            train_idx = split['train']
            test_idx = split['test']
            X = data['X']
            Y = data['Y']
            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_test = X[test_idx]
            y_test = Y[test_idx]
        else:
            print("No split found, using random 80/20...")
            X = data['X']
            Y = data['Y']
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            split_idx = int(len(X) * 0.8)
            X_train = X[:split_idx]
            y_train = Y[:split_idx]
            X_test = X[split_idx:]
            y_test = Y[split_idx:]

    # Flatten if needed
    if X_train.ndim > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"Train shape: {X_train.shape}")
    
    train_ds = SumoDataset(X_train, y_train)
    test_ds = SumoDataset(X_test, y_test)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)
    
    model = GNNLocalModule1(center_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        preds = []
        actuals = []
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                preds.append(out.cpu().numpy())
                actuals.append(by.squeeze(-1).cpu().numpy())
                
        preds = np.concatenate(preds)
        actuals = np.concatenate(actuals)
        
        r2 = r2_score(actuals, preds)
        mae = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        
        print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}, R2 {r2:.4f}, MAE {mae:.4f}, RMSE {rmse:.4f}")
        
    total_time = time.time() - start_time
    print(f"Final GNN (Module 1) Result: R2 {r2:.4f}, MAE {mae:.4f}, RMSE {rmse:.4f}, Time {total_time:.2f}s")

if __name__ == "__main__":
    train()
