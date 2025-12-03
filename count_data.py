import numpy as np

print("Loading data...")
try:
    d_orig = np.load("data/dataset_sumo_5km_lag12.npz", allow_pickle=True)
    d_filt = np.load("data/dataset_sumo_5km_lag12_filtered.npz", allow_pickle=True)

    print("\n" + "="*40)
    print("DATASET COUNTS")
    print("="*40)

    # Original Data
    n_orig = len(d_orig['Y'])
    n_zeros = (d_orig['Y'] == 0).sum()
    print(f"Original Total Samples: {n_orig:,}")
    print(f"Zero Values: {n_zeros:,} ({n_zeros/n_orig*100:.2f}%)")

    # Filtered Data
    n_filt = len(d_filt['Y'])
    print(f"Filtered Total Samples: {n_filt:,}")
    
    # Calculate removed
    n_removed = n_orig - n_filt
    print(f"Removed Samples: {n_removed:,} ({n_removed/n_orig*100:.2f}%)")

    print("\n" + "="*40)
    print("SPLIT ESTIMATES (Based on Paper)")
    print("="*40)
    
    # Paper says: train=953,351, val=119,168, test=119,168
    train_paper = 953351
    val_paper = 119168
    test_paper = 119168
    total_paper = train_paper + val_paper + test_paper
    
    print(f"Paper Total (Sum of splits): {total_paper:,}")
    print(f"Difference (Filtered - Paper Total): {n_filt - total_paper}")

except Exception as e:
    print(f"Error: {e}")
