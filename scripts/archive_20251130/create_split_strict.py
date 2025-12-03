import numpy as np
import json
from sklearn.model_selection import train_test_split

def create_split():
    npz_path = 'data/dataset_sumo_5km_lag12_filtered_strict.npz'
    split_path = 'data/sumo_split_strict.json'
    
    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    X = d['X']
    n_samples = len(X)
    
    print(f"Total samples: {n_samples}")
    
    indices = np.arange(n_samples)
    
    # 80% Train, 10% Val, 10% Test
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    
    split = {
        'train': train_idx.tolist(),
        'val': val_idx.tolist(),
        'test': test_idx.tolist()
    }
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    with open(split_path, 'w') as f:
        json.dump(split, f)
        
    print(f"Saved split to {split_path}")

if __name__ == "__main__":
    create_split()
