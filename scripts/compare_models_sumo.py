import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import argparse
import sys
import os

# Import model definitions
# We need to import the classes. Since they are in the scripts, we might need to duplicate them or import them.
# Importing from scripts is messy if they are not modules. I'll redefine them here for simplicity.

import torch.nn as nn
import torch.nn.functional as F

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

class DeepONet(nn.Module):
    def __init__(self, branch_dim=13, trunk_dim=10, latent_dim=128):
        super().__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        trunk_in = x[:, 1:11]
        branch_in = torch.cat([x[:, 0:1], x[:, 11:]], dim=1)
        b_out = self.branch(branch_in)
        t_out = self.trunk(trunk_in)
        out = torch.sum(b_out * t_out, dim=1, keepdim=True) + self.bias
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim=11, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 13, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, x):
        B = x.shape[0]
        context = x[:, 1:11]
        lags = x[:, 11:]
        current = x[:, 0:1]
        lags_flipped = torch.flip(lags, dims=[1])
        seq_vals = torch.cat([lags_flipped, current], dim=1).unsqueeze(-1)
        context_seq = context.unsqueeze(1).repeat(1, 13, 1)
        src = torch.cat([seq_vals, context_seq], dim=2)
        src = self.input_embedding(src) + self.pos_encoder
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        output = self.decoder(output)
        return output

class LSTMModel(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        B = x.shape[0]
        context = x[:, 1:11]
        lags = x[:, 11:]
        current = x[:, 0:1]
        lags_flipped = torch.flip(lags, dims=[1])
        seq_vals = torch.cat([lags_flipped, current], dim=1).unsqueeze(-1)
        context_seq = context.unsqueeze(1).repeat(1, 13, 1)
        src = torch.cat([seq_vals, context_seq], dim=2)
        output, _ = self.lstm(src)
        last_hidden = output[:, -1, :]
        out = self.fc(last_hidden)
        return out

class GNNLocal(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.w_self = nn.Linear(hidden_dim, hidden_dim)
        self.w_neighbor = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B = x.shape[0]
        center_feat = x[:, [0, 3]]
        up_feat = x[:, [7, 9]]
        down_feat = x[:, [8, 10]]
        h_c = F.relu(self.fc_in(center_feat))
        h_u = F.relu(self.fc_in(up_feat))
        h_d = F.relu(self.fc_in(down_feat))
        update = self.w_self(h_c) + self.w_neighbor(h_u + h_d)
        h_c_new = F.relu(update)
        h_c_new = self.dropout(h_c_new)
        out = self.fc_out(h_c_new)
        return out

def load_data():
    npz_path = 'data/dataset_sumo_5km_lag12_filtered_strict.npz'
    split_path = 'data/sumo_split_strict.json'
    
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    
    print(f"Loading split from {split_path}...")
    with open(split_path, 'r') as f:
        split = json.load(f)
        
    test_idx = split['test']
    train_idx = split['train'] # Needed for normalization stats
    
    return X, Y, train_idx, test_idx

def evaluate(model, X_test, Y_test, x_mean, x_std, y_mean, y_std, device):
    model.eval()
    batch_size = 1024
    preds = []
    trues = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            x_batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            y_batch = torch.FloatTensor(Y_test[i:i+batch_size]).to(device)
            
            x_batch = (x_batch - x_mean) / x_std
            
            pred = model(x_batch)
            pred = pred * y_std + y_mean
            
            preds.append(pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    r2 = r2_score(trues, preds)
    
    return mae, rmse, r2, preds, trues

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X, Y, train_idx, test_idx = load_data()
    
    # Stats
    x_train = X[train_idx]
    y_train = Y[train_idx]
    
    x_mean = torch.FloatTensor(x_train.mean(axis=0)).to(device)
    x_std = torch.FloatTensor(x_train.std(axis=0)).to(device)
    y_mean = torch.FloatTensor([y_train.mean()]).to(device)
    y_std = torch.FloatTensor([y_train.std()]).to(device)
    x_std[x_std < 1e-6] = 1.0
    
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    
    results = {}
    
    # MLP
    if os.path.exists('models/mlp_sumo_std.pth'):
        print("Evaluating MLP...")
        mlp = MLP(input_dim=23).to(device)
        mlp.load_state_dict(torch.load('models/mlp_sumo_std.pth'))
        mae, rmse, r2, _, _ = evaluate(mlp, X_test, Y_test, x_mean, x_std, y_mean, y_std, device)
        results['MLP'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"MLP: R2={r2:.4f}")
        
    # DeepONet
    if os.path.exists('models/deeponet_sumo_std.pth'):
        print("Evaluating DeepONet...")
        don = DeepONet(branch_dim=13, trunk_dim=10).to(device)
        don.load_state_dict(torch.load('models/deeponet_sumo_std.pth'))
        mae, rmse, r2, _, _ = evaluate(don, X_test, Y_test, x_mean, x_std, y_mean, y_std, device)
        results['DeepONet'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"DeepONet: R2={r2:.4f}")

    # Transformer
    if os.path.exists('models/transformer_sumo_std.pth'):
        print("Evaluating Transformer...")
        tf = TransformerModel().to(device)
        tf.load_state_dict(torch.load('models/transformer_sumo_std.pth'))
        mae, rmse, r2, _, _ = evaluate(tf, X_test, Y_test, x_mean, x_std, y_mean, y_std, device)
        results['Transformer'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"Transformer: R2={r2:.4f}")
        
    # LSTM
    if os.path.exists('models/lstm_sumo_std.pth'):
        print("Evaluating LSTM...")
        lstm = LSTMModel().to(device)
        lstm.load_state_dict(torch.load('models/lstm_sumo_std.pth'))
        mae, rmse, r2, _, _ = evaluate(lstm, X_test, Y_test, x_mean, x_std, y_mean, y_std, device)
        results['LSTM'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"LSTM: R2={r2:.4f}")
        
    # GNN
    if os.path.exists('models/gnn_sumo_std.pth'):
        print("Evaluating GNN...")
        gnn = GNNLocal().to(device)
        gnn.load_state_dict(torch.load('models/gnn_sumo_std.pth'))
        mae, rmse, r2, _, _ = evaluate(gnn, X_test, Y_test, x_mean, x_std, y_mean, y_std, device)
        results['GNN'] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"GNN: R2={r2:.4f}")
        
    print("\n=== Final Comparison ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
