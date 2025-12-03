import argparse
import numpy as np
import torch
import torch.nn as nn
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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 256, 256], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def load_data(npz_path, split_path):
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    
    # Features: 
    # 0: speed (t) - DO NOT USE for t+1 prediction if strictly avoiding leakage, but usually lags are t-11..t.
    # 1-6: Context (entered, left, density, occupancy, waitingTime, traveltime)
    # 7-10: Spatial
    # 11-22: Lags (t-11 ... t)
    
    # We use Lags (11-22) + Context (1-6) + Spatial (7-10)
    # We include index 0 (current speed) as it is the most important predictor.
    
    X_feat = X # Use all features
    
    print(f"Loading split from {split_path}...")
    with open(split_path, 'r') as f:
        split = json.load(f)
        
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    return (X_feat, Y), (train_idx, val_idx, test_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/dataset_sumo_5km_lag12_filtered_strict.npz')
    parser.add_argument('--split', default='data/sumo_split_strict.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1024) # Larger batch for larger dataset
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', default='models/mlp_sumo_std.pth')
    args = parser.parse_args()
    
    logger = StdLogger("MLP", "SUMO")
    logger.log_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    (X, Y), (train_idx, val_idx, test_idx) = load_data(args.npz, args.split)
    
    # Stats from Train
    x_train = X[train_idx]
    y_train = Y[train_idx]
    
    x_mean = torch.FloatTensor(x_train.mean(axis=0)).to(device)
    x_std = torch.FloatTensor(x_train.std(axis=0)).to(device)
    y_mean = torch.FloatTensor([y_train.mean()]).to(device)
    y_std = torch.FloatTensor([y_train.std()]).to(device)
    
    # Handle zero std
    x_std[x_std < 1e-6] = 1.0
    
    print(f"Train samples: {len(train_idx)}")
    print(f"Input dim: {X.shape[1]}")
    
    train_ds = SumoDataset(X[train_idx], Y[train_idx])
    val_ds = SumoDataset(X[val_idx], Y[val_idx])
    test_ds = SumoDataset(X[test_idx], Y[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    model = MLP(input_dim=X.shape[1]).to(device)
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
            print(f"Early stopping at epoch {epoch+1}")
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
