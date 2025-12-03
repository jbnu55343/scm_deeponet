import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, Dataset

class SumoDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class MLPSpatial(nn.Module):
    def __init__(self, input_size=23, hidden_size=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class DeepONetSpatial(nn.Module):
    def __init__(self, input_size=23, branch_depth=3, branch_width=256,
                 trunk_depth=3, trunk_width=256, output_width=256):
        super().__init__()
        branch_layers = []
        prev_size = input_size
        for i in range(branch_depth):
            branch_layers.append(nn.Linear(prev_size, branch_width))
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Dropout(0.1))
            prev_size = branch_width
        self.branch = nn.Sequential(*branch_layers)
        
        trunk_layers = []
        prev_size = 1
        for i in range(trunk_depth):
            trunk_layers.append(nn.Linear(prev_size, trunk_width))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(0.1))
            prev_size = trunk_width
        trunk_layers.append(nn.Linear(prev_size, output_width))
        self.trunk = nn.Sequential(*trunk_layers)
        
        self.output_linear = nn.Linear(output_width, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        branch_out = self.branch(x)
        dummy_input = torch.ones(batch_size, 1, device=x.device)
        trunk_out = self.trunk(dummy_input)
        combined = branch_out * trunk_out
        output = self.output_linear(combined)
        return output

def evaluate_model(model_path, model_class, test_loader, device):
    print(f"Evaluating {model_path}...")
    model = model_class().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load state dict: {e}")
        return
        
    model.eval()
    preds = []
    trues = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # Note: The original training script did NOT normalize inputs!
            # It passed X directly to the model.
            # See train_mlp_with_spatial.py: train_epoch just does out = model(X_batch)
            
            pred = model(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    # Handle shape mismatch if any
    if preds.ndim > 1: preds = preds.squeeze()
    if trues.ndim > 1: trues = trues.squeeze()
    
    mae = mean_absolute_error(trues, preds)
    r2 = r2_score(trues, preds)
    
    print(f"Results for {model_path}:")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    return r2

def main():
    npz_path = 'data/dataset_sumo_5km_lag12_filtered_with_spatial.npz'
    
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    
    # Temporal split 80/20
    n_train = int(len(X) * 0.8)
    X_test = X[n_train:]
    Y_test = Y[n_train:]
    
    print(f"Test set: {len(X_test)} samples")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_ds = SumoDataset(X_test, Y_test)
    test_loader = DataLoader(test_ds, batch_size=4096)
    
    evaluate_model('models/mlp_spatial.pt', lambda: MLPSpatial(input_size=23), test_loader, device)
    evaluate_model('models/deeponet_spatial.pt', lambda: DeepONetSpatial(input_size=23), test_loader, device)

if __name__ == '__main__':
    main()
