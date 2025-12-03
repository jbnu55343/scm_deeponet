import numpy as np
import json
import os

def create_split():
    npz_path = 'data/dataset_sumo_5km_lag12_nonzero.npz'
    output_path = 'data/sumo_split_nonzero.json'
    
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    Y = d['Y']
    n_samples = len(Y)
    
    indices = np.arange(n_samples)
    # Shuffle indices for random split
    np.random.seed(42)
    np.random.shuffle(indices)
    
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    print(f"Train: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")
    
    split = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(split, f)
        
    print(f"Saved split to {output_path}")

if __name__ == "__main__":
    create_split()
