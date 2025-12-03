import numpy as np
try:
    d = np.load('data/dataset_sumo_5km_lag12_filtered.npz')
    print(f"Shape: {d['X'].shape}")
except Exception as e:
    print(e)
