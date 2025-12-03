import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from scripts.std_utils import StdLogger, EarlyStopping

class SumoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class GNNLocal(nn.Module):
    def __init__(self, center_dim=19, neighbor_dim=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        # Center Node: Has full local history + features (19 dims)
        # Neighbor Nodes: Only have current speed/density (2 dims)
        
        self.fc_center = nn.Linear(center_dim, hidden_dim)
        self.fc_neighbor = nn.Linear(neighbor_dim, hidden_dim)
        
        # Graph Convolution (Simple Message Passing)
        # Node 0 (Center) receives from 1 (Up) and 2 (Down)
        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_neighbor = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, 23)
        # Features:
        # 0-6: Local current (speed, entered, left, density, occupancy, wait, travel)
        # 7-10: Spatial (speed_up, speed_down, density_up, density_down)
        # 11-22: Local Lags (speed_lag1..12)
        
        B = x.shape[0]
        
        # Center Features: Local Current (0-6) + Lags (11-22) -> 19 dims
        center_idx = [0, 1, 2, 3, 4, 5, 6] + list(range(11, 23))
        center_feat = x[:, center_idx]
        
        # Upstream: speed_up(7), density_up(9)
        up_feat = x[:, [7, 9]]
        
        # Downstream: speed_down(8), density_down(10)
        down_feat = x[:, [8, 10]]
        
        # Embed (B, H)
        h_c = F.relu(self.fc_center(center_feat))
        h_u = F.relu(self.fc_neighbor(up_feat))
        h_d = F.relu(self.fc_neighbor(down_feat))
        
        # Message Passing to Center
        # h_c' = W_self * h_c + W_neighbor * (h_u + h_d)
        
        update = self.w_self(h_c) + self.w_neighbor(h_u + h_d)
        h_c_new = F.relu(update)
        h_c_new = self.dropout(h_c_new)
        
        # Output
        out = self.fc_out(h_c_new)
        return out

def load_data(npz_path, split_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    
    with open(split_path, 'r') as f:
        split = json.load(f)
        
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    return (X, Y), (train_idx, val_idx, test_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/dataset_sumo_5km_lag12_filtered_strict.npz')
    parser.add_argument('--split', default='data/sumo_split_strict.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', default='models/gnn_sumo_std.pth')
    args = parser.parse_args()
    
    logger = StdLogger("GNN", "SUMO")
    logger.log_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    (X, Y), (train_idx, val_idx, test_idx) = load_data(args.npz, args.split)
    
    x_train = X[train_idx]
    y_train = Y[train_idx]
    
    x_mean = torch.FloatTensor(x_train.mean(axis=0)).to(device)
    x_std = torch.FloatTensor(x_train.std(axis=0)).to(device)
    y_mean = torch.FloatTensor([y_train.mean()]).to(device)
    y_std = torch.FloatTensor([y_train.std()]).to(device)
    x_std[x_std < 1e-6] = 1.0
    
    train_ds = SumoDataset(X[train_idx], Y[train_idx])
    val_ds = SumoDataset(X[val_idx], Y[val_idx])
    test_ds = SumoDataset(X[test_idx], Y[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    model = GNNLocal(center_dim=19, neighbor_dim=2, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std
            
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        preds = []
        trues = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = (x - x_mean) / x_std
                
                pred = model(x)
                pred = pred * y_std + y_mean
                
                y_norm = (y - y_mean) / y_std
                loss = criterion((pred - y_mean)/y_std, y_norm)
                val_loss += loss.item()
                
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())
                
        val_loss /= len(val_loader)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        r2 = r2_score(trues, preds)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save)
            logger.log_epoch(epoch+1, args.epochs, train_loss/len(train_loader), val_loss, r2)
        elif (epoch+1) % 5 == 0:
            logger.log_epoch(epoch+1, args.epochs, train_loss/len(train_loader), val_loss, r2)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Test
    model.load_state_dict(torch.load(args.save))
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = (x - x_mean) / x_std
            pred = model(x)
            pred = pred * y_std + y_mean
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    logger.log_result(trues, preds)

if __name__ == '__main__':
    main()
