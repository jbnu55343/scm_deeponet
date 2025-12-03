import numpy as np
import os

def filter_data():
    input_path = 'data/dataset_sumo_5km_lag12_filtered_with_spatial.npz'
    output_path = 'data/dataset_sumo_5km_lag12_nonzero.npz'
    
    print(f"Loading {input_path}...")
    d = np.load(input_path, allow_pickle=True)
    X = d['X']
    Y = d['Y']
    
    print(f"Original shape: X={X.shape}, Y={Y.shape}")
    
    # Filter condition: Keep samples where Feature 0 (Speed) > 0.1
    # We assume Feature 0 is speed.
    mask = X[:, 0] > 0.1
    
    X_new = X[mask]
    Y_new = Y[mask]
    
    print(f"Filtered shape: X={X_new.shape}, Y={Y_new.shape}")
    print(f"Removed {len(X) - len(X_new)} samples ({1 - len(X_new)/len(X):.2%})")
    
    print(f"Saving to {output_path}...")
    np.savez(output_path, X=X_new, Y=Y_new)
    print("Done.")

if __name__ == "__main__":
    filter_data()
