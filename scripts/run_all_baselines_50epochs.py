import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import sys
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Add current directory to path to allow imports if run from root
sys.path.append(os.getcwd())

# Try to import StdLogger, if not available, use a simple one
try:
    from scripts.std_utils import StdLogger, EarlyStopping
except ImportError:
    class StdLogger:
        def __init__(self, model_name, dataset_name):
            print(f"Initializing {model_name} on {dataset_name}")
        def log_config(self, args):
            print(f"Config: {args}")
        def log_epoch(self, epoch, total_epochs, train_loss, val_loss, val_r2):
            print(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val R2: {val_r2:.4f}")
        def log_result(self, trues, preds, time_taken=0):
            mse = mean_squared_error(trues, preds)
            mae = mean_absolute_error(trues, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(trues, preds)
            print(f"Final Results - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Time: {time_taken:.2f}s")
            return mse, mae, rmse, r2

    class EarlyStopping:
        def __init__(self, patience=10, min_delta=0.0001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0

# --- Data Loading ---
class SumoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def load_data(npz_path, split_path):
    print(f"Loading data from {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    
    print(f"Loading split from {split_path}...")
    with open(split_path, 'r') as f:
        split = json.load(f)
        
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    return (X, Y), (train_idx, val_idx, test_idx)

# --- Models ---

class PersistenceModel:
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        # Assuming column 0 is current speed
        return X[:, 0]

class MLP(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class LSTM(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        # Determine feature size for LSTM
        if input_dim == 23:
            # 1 lag (idx 0) + 12 lags (idx 11-22) = 13 sequence steps
            # 10 context (idx 1-10)
            self.seq_len = 13
            self.ctx_dim = 10
            self.lags_idx = [0] + list(range(11, 23))
            self.ctx_idx = list(range(1, 11))
        elif input_dim == 19:
            # 1 lag (idx 0) + 12 lags (idx 7-18) = 13 sequence steps
            # 6 context (idx 1-6)
            self.seq_len = 13
            self.ctx_dim = 6
            self.lags_idx = [0] + list(range(7, 19))
            self.ctx_idx = list(range(1, 7))
        else:
            # Fallback
            self.seq_len = 1
            self.ctx_dim = 0
            self.lags_idx = [0]
            self.ctx_idx = []
            
        # Input to LSTM: 1 (lag value) + ctx_dim
        self.lstm_input_dim = 1 + self.ctx_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (B, input_dim)
        B = x.size(0)
        
        # Extract Lags: (B, 13)
        lags = x[:, self.lags_idx]
        # Extract Context: (B, Ctx)
        ctx = x[:, self.ctx_idx]
        
        # Expand context to (B, 13, Ctx)
        ctx_expanded = ctx.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # Reshape lags to (B, 13, 1)
        lags_reshaped = lags.unsqueeze(2)
        
        # Concatenate: (B, 13, 1+Ctx)
        lstm_in = torch.cat([lags_reshaped, ctx_expanded], dim=2)
        
        # LSTM
        out, (hn, cn) = self.lstm(lstm_in)
        
        # Use last hidden state
        last_h = hn[-1]
        last_h = self.dropout(last_h)
        
        pred = self.fc(last_h)
        return pred

class TCN(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.fc = nn.Linear(hidden_dim * input_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim

    def forward(self, x):
        # x: (B, F) -> (B, 1, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GNNLocal(nn.Module):
    def __init__(self, center_dim=19, neighbor_dim=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.fc_center = nn.Linear(center_dim, hidden_dim)
        self.fc_neighbor = nn.Linear(neighbor_dim, hidden_dim)
        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_neighbor = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Expects 23 features
        # 0-6: Local current
        # 7-10: Spatial (up/down)
        # 11-22: Local Lags
        
        center_idx = [0, 1, 2, 3, 4, 5, 6] + list(range(11, 23))
        center_feat = x[:, center_idx]
        up_feat = x[:, [7, 9]]
        down_feat = x[:, [8, 10]]
        
        h_c = F.relu(self.fc_center(center_feat))
        h_u = F.relu(self.fc_neighbor(up_feat))
        h_d = F.relu(self.fc_neighbor(down_feat))
        
        update = self.w_self(h_c) + self.w_neighbor(h_u + h_d)
        h_c_new = F.relu(update)
        h_c_new = self.dropout(h_c_new)
        out = self.fc_out(h_c_new)
        return out

class DeepONet(nn.Module):
    def __init__(self, branch_dim=13, trunk_dim=10, latent_dim=128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.trunk_dim = trunk_dim
        
    def forward(self, x):
        # If trunk_dim is 10 (Spatial):
        # Trunk: 1-10
        # Branch: 0 + 11-22
        
        # If trunk_dim is 6 (No Spatial):
        # Trunk: 1-7
        # Branch: 0 + 7-19
        
        if self.trunk_dim == 10:
            trunk_in = x[:, 1:11]
            branch_in = torch.cat([x[:, 0:1], x[:, 11:]], dim=1)
        else:
            trunk_in = x[:, 1:7]
            branch_in = torch.cat([x[:, 0:1], x[:, 7:]], dim=1)
            
        b_out = self.branch(branch_in)
        t_out = self.trunk(trunk_in)
        return torch.sum(b_out * t_out, dim=1, keepdim=True) + self.bias

class TransformerModel(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, hidden_dim) # Embed each feature
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim * input_dim, 1)
        self.input_dim = input_dim

    def forward(self, x):
        # x: (B, F) -> (B, F, 1) -> (B, F, H)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.size(0), -1)
        out = self.fc_out(x)
        return out

# --- Training Wrapper ---

def train_torch_model(model, train_loader, val_loader, test_loader, device, epochs=50, lr=0.001, name="Model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    
    print(f"\nTraining {name}...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            
        early_stopping(val_loss)
        if (epoch+1) % 1 == 0:
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    train_time = time.time() - start_time
    
    # Test
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    return trues, preds, train_time

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/dataset_sumo_5km_lag12_filtered_with_spatial.npz')
    parser.add_argument('--split', default='data/sumo_split_standard.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--out', default='results_50epochs.txt')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    (X, Y), (train_idx, val_idx, test_idx) = load_data(args.npz, args.split)
    
    # Check dimensions
    input_dim = X.shape[1]
    print(f"Input dimension: {input_dim}")
    if input_dim != 23 and input_dim != 19:
        print(f"WARNING: Expected 23 or 19 features, got {input_dim}. GNN and DeepONet might fail.")
    
    # Split
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Standardization
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)
    
    # Target Standardization (for Neural Nets)
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    Y_train_scaled = (Y_train - y_mean) / y_std
    Y_val_scaled = (Y_val - y_mean) / y_std
    Y_test_scaled = (Y_test - y_mean) / y_std
    
    # DataLoaders
    train_ds = SumoDataset(X_train_scaled, Y_train_scaled)
    val_ds = SumoDataset(X_val_scaled, Y_val_scaled)
    test_ds = SumoDataset(X_test_scaled, Y_test_scaled)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    results = {}
    
    # 1. Persistence
    print("\n--- Running Persistence ---")
    start_time = time.time()
    # Persistence uses raw X (unscaled) or scaled? 
    # Usually raw. But we need to be careful.
    # Persistence: y_pred = x_speed.
    # If we use scaled X, we get scaled speed.
    # If we compare to scaled Y, it might work if mean/std are similar.
    # Safer to use raw values.
    p_model = PersistenceModel()
    p_preds = p_model.predict(X_test)
    p_time = time.time() - start_time
    
    mse = mean_squared_error(Y_test, p_preds)
    mae = mean_absolute_error(Y_test, p_preds)
    r2 = r2_score(Y_test, p_preds)
    results['Persistence'] = {'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse), 'R2': r2, 'Time': p_time}
    print(f"Persistence: R2={r2:.4f}, RMSE={np.sqrt(mse):.4f}")
    
    # 2. Ridge
    print("\n--- Running Ridge ---")
    start_time = time.time()
    ridge = Ridge()
    ridge.fit(X_train_scaled, Y_train) # Train on scaled X, raw Y
    r_preds = ridge.predict(X_test_scaled)
    r_time = time.time() - start_time
    
    mse = mean_squared_error(Y_test, r_preds)
    mae = mean_absolute_error(Y_test, r_preds)
    r2 = r2_score(Y_test, r_preds)
    results['Ridge'] = {'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse), 'R2': r2, 'Time': r_time}
    print(f"Ridge: R2={r2:.4f}, RMSE={np.sqrt(mse):.4f}")
    
    # Helper for NN evaluation
    def evaluate_nn(name, model_class, **kwargs):
        model = model_class(**kwargs).to(device)
        trues_scaled, preds_scaled, train_time = train_torch_model(
            model, train_loader, val_loader, test_loader, device, epochs=args.epochs, name=name
        )
        # Inverse transform
        preds = preds_scaled * y_std + y_mean
        trues = trues_scaled * y_std + y_mean
        
        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds)
        results[name] = {'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse), 'R2': r2, 'Time': train_time}
        print(f"{name}: R2={r2:.4f}, RMSE={np.sqrt(mse):.4f}")

    # 3. MLP
    evaluate_nn("MLP", MLP, input_dim=input_dim)
    
    # 4. TCN
    evaluate_nn("TCN", TCN, input_dim=input_dim)
    
    # 5. GNN
    if input_dim == 23:
        evaluate_nn("GNN", GNNLocal)
    else:
        print("Skipping GNN for non-spatial dataset (dim != 23)")
    
    # 6. DeepONet
    if input_dim == 23:
        evaluate_nn("DeepONet", DeepONet, branch_dim=13, trunk_dim=10)
    elif input_dim == 19:
        evaluate_nn("DeepONet", DeepONet, branch_dim=13, trunk_dim=6)
    
    # 7. Transformer
    evaluate_nn("Transformer", TransformerModel, input_dim=input_dim)

    # 8. LSTM
    evaluate_nn("LSTM", LSTM, input_dim=input_dim)
    
    # Save Results
    print(f"\nSaving results to {args.out}...")
    with open(args.out, 'w') as f:
        f.write(f"{'Model':<15} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R2':<10} {'Time':<10}\n")
        f.write("-" * 70 + "\n")
        for name, metrics in results.items():
            f.write(f"{name:<15} {metrics['MSE']:.4f}     {metrics['MAE']:.4f}     {metrics['RMSE']:.4f}     {metrics['R2']:.4f}     {metrics['Time']:.2f}\n")
            
    print("Done.")

if __name__ == '__main__':
    main()
