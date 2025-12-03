import numpy as np
import json
import os

def main():
    data_path = "data/dataset_sumo_5km_lag12_filtered_with_spatial.npz"
    split_path = "data/sumo_split_standard.json"
    
    print(f"Loading {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    n_samples = len(data['Y'])
    print(f"Total samples: {n_samples}")
    
    # Temporal Split: 70% Train, 10% Val, 20% Test
    n_train = int(n_samples * 0.70)
    n_val = int(n_samples * 0.10)
    n_test = n_samples - n_train - n_val
    
    indices = np.arange(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    print(f"Train: {len(train_idx)} ({len(train_idx)/n_samples:.2%})")
    print(f"Val:   {len(val_idx)} ({len(val_idx)/n_samples:.2%})")
    print(f"Test:  {len(test_idx)} ({len(test_idx)/n_samples:.2%})")
    
    split = {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist()
    }
    
    with open(split_path, 'w') as f:
        json.dump(split, f)
    
    print(f"Saved split to {split_path}")

if __name__ == '__main__':
    main()
