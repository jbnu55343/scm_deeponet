import argparse
import numpy as np
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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

def load_data(npz_path):
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    # Random Split 80/10/10
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    
    return (X, Y), (train_idx, val_idx, test_idx)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/dataset_sumo_5km_lag12_filtered_strict.npz')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', default='models/mlp_sumo_strict.pth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    (X, Y), (train_idx, val_idx, test_idx) = load_data(args.npz)
    
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
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Normalize
            x = (x - x_mean) / x_std
            y = (y - y_mean) / y_std
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                x = (x - x_mean) / x_std
                y = (y - y_mean) / y_std
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save)
            
    # Test
    model.load_state_dict(torch.load(args.save))
    model.eval()
    preds = []
    targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = (x - x_mean) / x_std
            # Do NOT normalize target for metric calculation (we want real scale)
            # But model predicts normalized
            pred = model(x)
            pred = pred * y_std + y_mean
            
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    
    print(f"\nTest Results on Strict Filtered Data:")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Calculate Persistence R2 on Test Set
    # Assuming feature 0 is speed
    # We need to get X_test unnormalized
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    
    # Persistence: Predict X[:, 0] as Y
    y_persist = X_test[:, 0]
    r2_persist = r2_score(Y_test, y_persist)
    print(f"Persistence R2: {r2_persist:.4f}")

if __name__ == "__main__":
    main()
