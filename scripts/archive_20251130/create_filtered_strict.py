#!/usr/bin/env python3
"""
Create strictly filtered dataset where BOTH X (current speed) and Y (next speed) are > 0.
This avoids the distribution mismatch caused by filtering only Y > 0.
"""

import numpy as np
import os

def create_strict_filtered_dataset():
    input_path = "data/dataset_sumo_5km_lag12.npz"
    output_path = "data/dataset_sumo_5km_lag12_filtered_strict.npz"
    
    print(f"Loading {input_path}...")
    d = np.load(input_path, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    features = d['features']
    meta = d['meta'].item()
    
    print(f"Original shape: X={X.shape}, Y={Y.shape}")
    
    # Find speed index (usually 0, but let's be safe)
    # features is a numpy array of strings
    feat_list = list(features)
    speed_idx = feat_list.index('speed')
    print(f"Speed feature index: {speed_idx}")
    
    # Filter: X[speed] > 0 AND Y > 0
    # Y is (N, 1)
    mask = (X[:, speed_idx] > 0) & (Y.reshape(-1) > 0)
    
    X_f = X[mask]
    Y_f = Y[mask]
    
    print(f"Filtered shape: X={X_f.shape}, Y={Y_f.shape}")
    print(f"Retention rate: {len(X_f)/len(X)*100:.2f}%")
    
    print(f"Y Mean: {Y_f.mean():.4f}")
    print(f"X Mean: {X_f[:, speed_idx].mean():.4f}")
    
    # Save
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        X=X_f,
        Y=Y_f,
        features=features,
        target=d['target'],
        meta=meta
    )
    print("Done.")

if __name__ == "__main__":
    create_strict_filtered_dataset()
