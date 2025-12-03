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

# --- Data Loading (Adapted from train_mlp_metr_la.py) ---
def load_npz(path):
    print(f"Loading {path}...")
    d = np.load(path, allow_pickle=True, mmap_mode='r')
    X = d["X"]
    Y = d["Y"]
    
    # Handle features
    if "features" in d.files:
        feats = d["features"].tolist()
    else:
        feats = []

    # Handle split
    split = None
    if "split" in d.files:
        try:
            split = d["split"].item()
        except:
            split = None
            
    return X, Y, feats, split

class MetrLaDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        # Ensure Y is (N, 1)
        self.Y = torch.FloatTensor(Y).view(-1, 1)
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# --- Models ---

class PersistenceModel:
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        # Assuming column 0 is the most recent lag (t-1)
        # Check feature list if possible, but standard convention is lag1 at idx 0
        return X[:, 0]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
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
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        # Revert to treating input as single step (MLP-like LSTM)
        # This is safer when feature layout is unknown
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (B, input_dim)
        # Reshape to (B, 1, input_dim)
        x = x.unsqueeze(1)
        
        out, (hn, cn) = self.lstm(x)
        
        # Use last hidden state
        last_h = hn[-1]
        last_h = self.dropout(last_h)
        
        pred = self.fc(last_h)
        return pred

class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, kernel_size=3, dropout=0.1):
        super().__init__()
        # Treat input as a sequence of features (1D Conv over features)
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

class DeepONet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=128, dropout=0.1):
        super().__init__()
        # Assuming input_dim = 12 * 207 = 2484
        # We split into History (Lags 1-11) and Current/Context (Lag 12)
        self.num_nodes = 207
        self.seq_len = 12
        
        self.branch_dim = (self.seq_len - 1) * self.num_nodes # 11 * 207 = 2277
        self.trunk_dim = self.num_nodes # 1 * 207 = 207
        
        # Fallback if input_dim doesn't match expected
        if input_dim != 2484:
            print(f"Warning: DeepONet expected 2484 inputs, got {input_dim}. Using random split.")
            self.branch_dim = input_dim // 2
            self.trunk_dim = input_dim - self.branch_dim

        self.branch = nn.Sequential(
            nn.Linear(self.branch_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.trunk = nn.Sequential(
            nn.Linear(self.trunk_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # x: (B, input_dim)
        B = x.size(0)
        
        if x.shape[1] == 2484:
            # Split into History (Branch) and Current (Trunk)
            # Assuming features are ordered by time: [t-11, ..., t]
            # Actually, usually it's [node1_t-11, node2_t-11... node1_t, ...] or [t-11_all, t-10_all...]
            # Let's assume [t-11_all, ..., t_all]
            split_idx = self.branch_dim
            b_in = x[:, :split_idx]
            t_in = x[:, split_idx:]
        else:
            b_in = x[:, :self.branch_dim]
            t_in = x[:, self.branch_dim:]
        
        b_out = self.branch(b_in)
        t_out = self.trunk(t_in)
        
        # Dot product
        return torch.sum(b_out * t_out, dim=1, keepdim=True) + self.bias

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        # Reshape input to (B, Seq_Len, Feat_Dim)
        # Assuming 12 steps, 207 nodes
        self.seq_len = 12
        self.feat_dim = 207
        
        if input_dim != 2484:
             # Fallback
             self.seq_len = 1
             self.feat_dim = input_dim
        
        self.embedding = nn.Linear(self.feat_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Flatten output of transformer: (B, Seq_Len * Hidden)
        self.fc_out = nn.Linear(self.seq_len * hidden_dim, 1)

    def forward(self, x):
        # x: (B, 2484)
        B = x.size(0)
        
        if x.shape[1] == 2484:
            x = x.view(B, self.seq_len, self.feat_dim) # (B, 12, 207)
        else:
            x = x.unsqueeze(1) # (B, 1, F)
            
        x = self.embedding(x) # (B, 12, H)
        x = self.transformer_encoder(x)
        x = x.reshape(B, -1) # (B, 12*H)
        out = self.fc_out(x)
        return out

# --- Training & Eval ---

def train_torch_model(model, train_loader, val_loader, test_loader, device, epochs=50, lr=0.001, name="Model"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\nTraining {name}...")
    start_train = time.time()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Debug shape on first batch of first epoch
            if epoch == 0 and i == 0:
                print(f"DEBUG: x.shape={x.shape}, y.shape={y.shape}")
                
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
            counter = 0
        else:
            counter += 1
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    train_time = time.time() - start_train
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # Inference Speed Test
    model.eval()
    start_inf = time.time()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    end_inf = time.time()
    inference_time = end_inf - start_inf
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    return trues, preds, train_time, inference_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/metr_la_lag12_temporal.npz')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--out', default='results_metr_la_baselines.txt')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    X, Y, feats, split = load_npz(args.npz)
    
    # Handle Split
    if split is not None:
        train_idx = split['train']
        val_idx = split['val']
        test_idx = split['test']
    else:
        print("No split found, using random split 70/15/15")
        N = len(X)
        indices = np.arange(N)
        np.random.shuffle(indices)
        train_idx = indices[:int(0.7*N)]
        val_idx = indices[int(0.7*N):int(0.85*N)]
        test_idx = indices[int(0.85*N):]
        
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Standardization
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)
    
    y_mean = Y_train.mean()
    y_std = Y_train.std()
    
    Y_train_scaled = (Y_train - y_mean) / y_std
    Y_val_scaled = (Y_val - y_mean) / y_std
    Y_test_scaled = (Y_test - y_mean) / y_std
    
    # DataLoaders
    train_ds = MetrLaDataset(X_train_scaled, Y_train_scaled)
    val_ds = MetrLaDataset(X_val_scaled, Y_val_scaled)
    test_ds = MetrLaDataset(X_test_scaled, Y_test_scaled)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    results = {}
    input_dim = X.shape[1]
    
    # 1. Persistence
    # print("\n--- Running Persistence ---")
    # start_inf = time.time()
    # p_model = PersistenceModel()
    # # Use raw X for persistence to be safe, or scaled X and compare to scaled Y
    # # Let's use raw X and raw Y for persistence to avoid scaling artifacts
    # p_preds = p_model.predict(X_test)
    # end_inf = time.time()
    
    # mse = mean_squared_error(Y_test, p_preds)
    # r2 = r2_score(Y_test, p_preds)
    # results['Persistence'] = {
    #     'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mean_absolute_error(Y_test, p_preds), 'R2': r2, 
    #     'TrainTime': 0, 'InfTime': end_inf - start_inf
    # }
    # print(f"Persistence: R2={r2:.4f}")
    
    # 2. Ridge
    # print("\n--- Running Ridge ---")
    # start_train = time.time()
    # ridge = Ridge()
    # ridge.fit(X_train_scaled, Y_train)
    # train_time = time.time() - start_train
    
    # start_inf = time.time()
    # r_preds = ridge.predict(X_test_scaled)
    # end_inf = time.time()
    
    # mse = mean_squared_error(Y_test, r_preds)
    # r2 = r2_score(Y_test, r_preds)
    # results['Ridge'] = {
    #     'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mean_absolute_error(Y_test, r_preds), 'R2': r2, 
    #     'TrainTime': train_time, 'InfTime': end_inf - start_inf
    # }
    # print(f"Ridge: R2={r2:.4f}")
    
    # Helper for NN
    def evaluate_nn(name, model_class, **kwargs):
        print(f"\nPreparing {name}...", flush=True)
        try:
            print(f"Initializing {name} model...", flush=True)
            model = model_class(**kwargs).to(device)
            print(f"Model initialized. Starting training...", flush=True)
            trues_scaled, preds_scaled, train_time, inf_time = train_torch_model(
                model, train_loader, val_loader, test_loader, device, epochs=args.epochs, name=name
            )
            
            # Inverse transform
            preds = preds_scaled * y_std + y_mean
            trues = trues_scaled * y_std + y_mean
            
            mse = mean_squared_error(trues, preds)
            r2 = r2_score(trues, preds)
            results[name] = {
                'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mean_absolute_error(trues, preds), 'R2': r2, 
                'TrainTime': train_time, 'InfTime': inf_time
            }
            print(f"{name}: R2={r2:.4f}, InfTime={inf_time:.4f}s", flush=True)
        except Exception as e:
            print(f"Error running {name}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # 3. MLP
    evaluate_nn("MLP", MLP, input_dim=input_dim)

    # 4. LSTM
    evaluate_nn("LSTM", LSTM, input_dim=input_dim)
    
    # 5. TCN
    # evaluate_nn("TCN", TCN, input_dim=input_dim)

    # 6. DeepONet
    # evaluate_nn("DeepONet", DeepONet, input_dim=input_dim)
    
    # 7. Transformer
    # evaluate_nn("Transformer", TransformerModel, input_dim=input_dim)
    
    # Save
    with open(args.out, 'w') as f:
        f.write(f"{'Model':<15} {'MSE':<10} {'MAE':<10} {'RMSE':<10} {'R2':<10} {'TrainTime':<10} {'InfTime':<10}\n")
        f.write("-" * 85 + "\n")
        for name, m in results.items():
            f.write(f"{name:<15} {m['MSE']:.4f}     {m['MAE']:.4f}     {m['RMSE']:.4f}     {m['R2']:.4f}     {m['TrainTime']:.2f}       {m['InfTime']:.4f}\n")
            
    print(f"\nResults saved to {args.out}")

if __name__ == '__main__':
    main()
