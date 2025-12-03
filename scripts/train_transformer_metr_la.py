import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score

class MetrLaTemporalDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X) # (N, Lag)
        self.Y = torch.FloatTensor(Y).unsqueeze(1) # (N, 1)
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_dim=207, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model)) # Max len 100
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, 1)
        
    def forward(self, src):
        # src: (B, 2484) -> (B, 12, 207)
        B = src.shape[0]
        src = src.view(B, 12, 207)
        
        # Embedding
        src = self.input_embedding(src) # (B, 12, d_model)
        
        # Positional Encoding
        B, L, D = src.shape
        src = src + self.pos_encoder[:, :L, :]
        
        # Transformer
        output = self.transformer_encoder(src) # (B, 12, d_model)
        
        # Pooling (Use last time step)
        output = output[:, -1, :] # (B, d_model)
        
        # Decode
        output = self.decoder(output) # (B, 1)
        return output

def load_data(npz_path):
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    X = d['X'] # (N, Lag)
    Y = d['Y'] # (N,)
    
    # Metadata
    meta = d['meta'].item() if 'meta' in d else {}
    orig_std = float(meta.get('norm_std', 1.0))
    orig_mean = float(meta.get('norm_mean', 0.0))
    
    # Split
    split = d['split'].item()
    train_idx = split['train']
    val_idx = split['val']
    test_idx = split['test']
    
    return (X, Y), (train_idx, val_idx, test_idx), (orig_mean, orig_std)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', default='data/metr_la_lag12_temporal.npz')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save', default='models/transformer_metr_la.pth')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Data
    (X, Y), (train_idx, val_idx, test_idx), (orig_mean, orig_std) = load_data(args.npz)
    
    # Calculate Stats for Normalization (on the fly)
    x_train = X[train_idx]
    y_train = Y[train_idx]
    
    x_mean = torch.FloatTensor([x_train.mean()]).to(device)
    x_std = torch.FloatTensor([x_train.std()]).to(device)
    y_mean = torch.FloatTensor([y_train.mean()]).to(device)
    y_std = torch.FloatTensor([y_train.std()]).to(device)
    
    print(f"Y Mean: {y_mean.item():.2f}, Y Std: {y_std.item():.2f}")
    
    # Datasets
    train_ds = MetrLaTemporalDataset(X[train_idx], Y[train_idx])
    val_ds = MetrLaTemporalDataset(X[val_idx], Y[val_idx])
    test_ds = MetrLaTemporalDataset(X[test_idx], Y[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    
    # Model
    model = TransformerModel(input_dim=207, d_model=64, nhead=4, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Normalize
            x = (x - x_mean) / (x_std + 1e-8)
            y = (y - y_mean) / (y_std + 1e-8)
            
            pred = model(x)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        preds = []
        trues = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                # Normalize Input
                x = (x - x_mean) / (x_std + 1e-8)
                
                pred = model(x)
                
                # Denormalize Output
                pred = pred * y_std + y_mean
                
                # Calculate Loss on Normalized (for consistency) or Real?
                # Let's use Real for metrics, Normalized for Loss tracking
                # But wait, criterion expects normalized y if we trained on normalized y.
                # Let's stick to normalized loss for tracking.
                y_norm = (y - y_mean) / (y_std + 1e-8)
                loss = criterion(pred, y) # Wait, pred is denormalized now? No, pred is normalized output.
                # Re-do:
                # pred_norm = model(x)
                # loss = criterion(pred_norm, y_norm)
                # val_loss += loss.item()
                # pred_real = pred_norm * y_std + y_mean
                
                # Correct logic:
                pred_norm = model(x)
                y_norm = (y - y_mean) / (y_std + 1e-8)
                loss = criterion(pred_norm, y_norm)
                val_loss += loss.item()
                
                pred_real = pred_norm * y_std + y_mean
                preds.append(pred_real.cpu().numpy())
                trues.append(y.cpu().numpy())
                
        val_loss /= len(val_loader)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        
        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save)
            print(f"Epoch {epoch+1:03d} | Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.2f} | R2: {r2:.4f} *")
        elif (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.2f} | R2: {r2:.4f}")
            
    training_time = time.time() - start_time
    
    # Final Test
    model.load_state_dict(torch.load(args.save))
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = (x - x_mean) / (x_std + 1e-8)
            pred = model(x)
            pred = pred * y_std + y_mean
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    test_mae = mean_absolute_error(trues, preds)
    test_rmse = np.sqrt(((preds - trues)**2).mean())
    test_r2 = r2_score(trues, preds)
    
    print(f"\n[TEST] MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f} | R2: {test_r2:.4f}")
    
    # Standardized Output
    num_params = sum(p.numel() for p in model.parameters())
    result = {
        "model": "Transformer",
        "params": num_params,
        "time_sec": training_time,
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2)
    }
    print("FINAL_RESULT_JSON:" + json.dumps(result))

if __name__ == '__main__':
    main()
