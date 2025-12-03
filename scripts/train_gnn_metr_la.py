import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import time

class MetrLaFullDataset(Dataset):
    def __init__(self, data, lag=12, horizon=1):
        self.data = torch.FloatTensor(data)
        self.lag = lag
        self.horizon = horizon
        self.n_samples = self.data.shape[0] - lag - horizon + 1
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # X: (Lag, Nodes)
        # Y: (Nodes,)
        x = self.data[idx : idx+self.lag, :]
        y = self.data[idx+self.lag+self.horizon-1, :]
        return x, y

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: (B, Nodes, In)
        # adj: (Nodes, Nodes)
        
        # Support: (B, Nodes, Out)
        support = self.linear(x)
        
        # Aggregation: A * Support
        # (Nodes, Nodes) @ (B, Nodes, Out) -> (B, Nodes, Out)
        # We need to permute support to (B, Out, Nodes) for matmul or expand A
        
        # Easier: (B, Nodes, Out) -> (B, Out, Nodes)
        # But A is (N, N).
        # Let's use bmm with expanded A
        
        B, N, C = support.shape
        
        # A_batch: (B, N, N)
        A_batch = adj.unsqueeze(0).expand(B, -1, -1)
        
        # Out: (B, N, N) @ (B, N, C) -> (B, N, C)
        out = torch.bmm(A_batch, support)
        
        return out

class GCN(nn.Module):
    def __init__(self, num_nodes, lag, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.num_nodes = num_nodes
        self.lag = lag
        
        # Input is (B, Lag, Nodes) -> Permute to (B, Nodes, Lag)
        # So input features = Lag
        
        self.gc1 = GCNLayer(lag, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        
        # Learnable Adjacency? Or computed from data?
        # We'll use a parameter for now to let it learn the graph structure
        # (Adaptive Graph)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        
    def forward(self, x):
        # x: (B, Lag, Nodes)
        B, L, N = x.shape
        
        # Permute to (B, Nodes, Lag)
        x = x.permute(0, 2, 1)
        
        # Compute Adjacency from Embeddings
        # A = Softmax(ReLU(E @ E.T))
        adj = F.relu(torch.mm(self.node_embeddings, self.node_embeddings.t()))
        adj = F.softmax(adj, dim=1)
        
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = self.fc(x) # (B, N, 1)
        
        return x.squeeze(-1) # (B, N)

def load_data(h5_path):
    print(f"Loading {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Keys might vary. Usually 'df' -> 'block0_values'
        # Let's inspect keys if needed. Assuming standard structure.
        # Based on previous context, it has 'df' group
        if 'df' in f:
            data = f['df']['block0_values'][:]
        else:
            # Try to find the dataset
            key = list(f.keys())[0]
            data = f[key][:]
            
    print(f"Data shape: {data.shape}")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='data/METR-LA.h5')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    data = load_data(args.h5)
    
    # Normalize
    mean = data.mean()
    std = data.std()
    data = (data - mean) / std
    print(f"Normalized: mean={mean:.2f}, std={std:.2f}")
    
    # Split
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.1)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    train_ds = MetrLaFullDataset(train_data)
    val_ds = MetrLaFullDataset(val_data)
    test_ds = MetrLaFullDataset(test_data)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    # Model
    model = GCN(num_nodes=data.shape[1], lag=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    start_train_time = time.time()
    for epoch in range(args.epochs):
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
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    
    training_time_total = time.time() - start_train_time
        
    # Test
    model.eval()
    preds = []
    trues = []
    start_time = time.time()
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # Denormalize
            pred = pred * std + mean
            y = y * std + mean
            
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    # Evaluate ONLY on Node 0 to be comparable with MLP/DeepONet
    # MLP/DeepONet were trained/tested on Node 0 specifically.
    # GNN predicts all nodes, but we should only count the error on Node 0 for fair comparison.
    target_node = 0
    preds_node0 = preds[:, target_node]
    trues_node0 = trues[:, target_node]
    
    mae = np.mean(np.abs(preds_node0 - trues_node0))
    rmse = np.sqrt(np.mean((preds_node0 - trues_node0)**2))
    r2 = 1 - np.sum((preds_node0 - trues_node0)**2) / np.sum((trues_node0 - trues_node0.mean())**2)
    
    end_time = time.time()
    training_time = end_time - start_time # This is inference time, wait. I need training time.
    
    print(f"Test MAE (Node 0): {mae:.2f}")
    print(f"Test RMSE (Node 0): {rmse:.2f}")
    print(f"Test R2 (Node 0): {r2:.4f}")
    
    # Standardized Output
    import json
    num_params = sum(p.numel() for p in model.parameters())
    result = {
        "model": "GNN (GCN)",
        "params": num_params,
        "time_sec": training_time_total,
        "test_mae": float(mae),
        "test_rmse": float(rmse),
        "test_r2": float(r2)
    }
    print("FINAL_RESULT_JSON:" + json.dumps(result))

if __name__ == '__main__':
    main()
