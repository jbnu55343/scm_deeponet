import numpy as np

try:
    d = np.load('data/metr_la_lag12_temporal.npz', allow_pickle=True)
    print(f"Keys: {list(d.keys())}")
    if 'features' in d:
        feats = d['features']
        print(f"Features shape: {feats.shape}")
        print(f"First 20 features: {feats[:20]}")
        print(f"Last 20 features: {feats[-20:]}")
    else:
        print("No 'features' key found.")
        
    X = d['X']
    print(f"X shape: {X.shape}")
    
except Exception as e:
    print(f"Error: {e}")
