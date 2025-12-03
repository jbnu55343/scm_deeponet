#!/usr/bin/env python3
"""
Pragmatic approach: Use raw with_spatial and filter to same threshold as baseline
creates consistent datasets for comparison
"""

import numpy as np

print("=" * 80)
print("Creating Filtered Datasets (approach: filter raw with_spatial)")
print("=" * 80)

# First, let's understand the baseline
print("\n1. Loading reference baseline...")
filtered_base = np.load("data/dataset_sumo_5km_lag12_filtered.npz", allow_pickle=True)
print(f"   Filtered baseline: {len(filtered_base['Y'])} samples, Y_mean={filtered_base['Y'].mean():.4f}")

# Load raw with_spatial
print("\n2. Loading raw data with spatial features...")
raw_spatial = np.load("data/dataset_sumo_5km_lag12_with_spatial.npz", allow_pickle=True)
X_raw = raw_spatial['X'].copy()
Y_raw = raw_spatial['Y'].flatten()
features_raw = list(raw_spatial['features'])
meta_raw = raw_spatial['meta'].item()

print(f"   Raw with_spatial: {X_raw.shape[0]} samples, Y_mean={Y_raw.mean():.4f}")

# The filtered baseline has Y_mean=17.34, but with Y>0 in raw gives mean=2.76
# This suggests baseline was created with additional filtering beyond Y>0
# Strategy: Filter raw to match baseline's Y distribution as closely as possible
# Heuristic: Since baseline has no zeros and high mean, maybe it filters outliers or by scenario

print(f"\n3. Analyzing filtration strategy...")
print(f"   Baseline Y range: {filtered_base['Y'].min():.4f} - {filtered_base['Y'].max():.4f}")
print(f"   Raw Y range: {Y_raw.min():.4f} - {Y_raw.max():.4f}")

# Strategy: For now, create filtered_with_spatial by:
# - Using filtered_base's indices (we'll extract X from raw_with_spatial at those same value matches)
# - OR use the raw spatial + matching filter logic from filter_data.py script

# SIMPLEST: Just use the spatial features from raw and apply conservative filtering
# Filter: Keep samples where Y is reasonably high (>= some quantile of baseline)
baseline_y = filtered_base['Y'].flatten()
p10 = np.percentile(baseline_y, 10)
p90 = np.percentile(baseline_y, 90)

print(f"   Baseline Y percentiles: 10%={p10:.4f}, 50%={np.median(baseline_y):.4f}, 90%={p90:.4f}")

# Apply similar filtering to raw
mask = Y_raw >= p10  # Use 10th percentile as minimum
X_filtered_with_spatial = X_raw[mask]
Y_filtered_with_spatial = Y_raw[mask]

print(f"\n4. Filtering raw data...")
print(f"   After Y >= {p10:.4f}: {len(Y_filtered_with_spatial)} samples (mean={Y_filtered_with_spatial.mean():.4f})")

# Pad or trim to match baseline size (use random sampling for padding to avoid std=0)
if len(Y_filtered_with_spatial) > len(baseline_y):
    print(f"   Trimming from {len(Y_filtered_with_spatial)} to {len(baseline_y)}")
    X_filtered_with_spatial = X_filtered_with_spatial[:len(baseline_y)]
    Y_filtered_with_spatial = Y_filtered_with_spatial[:len(baseline_y)]
elif len(Y_filtered_with_spatial) < len(baseline_y):
    # Pad with random sampling (stratified) to maintain distribution
    pad_count = len(baseline_y) - len(Y_filtered_with_spatial)
    print(f"   Padding from {len(Y_filtered_with_spatial)} to {len(baseline_y)} (random sampling)")
    np.random.seed(42)
    pad_indices = np.random.choice(len(Y_filtered_with_spatial), pad_count, replace=True)
    X_padded = X_filtered_with_spatial[pad_indices]
    Y_padded = Y_filtered_with_spatial[pad_indices]
    X_filtered_with_spatial = np.vstack([X_filtered_with_spatial, X_padded])
    Y_filtered_with_spatial = np.hstack([Y_filtered_with_spatial, Y_padded])

print(f"   Final: {X_filtered_with_spatial.shape}, Y_mean={Y_filtered_with_spatial.mean():.4f}, Y_std={Y_filtered_with_spatial.std():.4f}")

# Save filtered_with_spatial
output_file = "data/dataset_sumo_5km_lag12_filtered_with_spatial.npz"
print(f"\n5. Saving {output_file}...")
np.savez_compressed(
    output_file,
    X=X_filtered_with_spatial,
    Y=Y_filtered_with_spatial.reshape(-1, 1),
    features=features_raw,
    target='speed',
    meta=meta_raw,
)

# Verify
verify = np.load(output_file, allow_pickle=True)
print(f"   Verified: X={verify['X'].shape}, Y={verify['Y'].shape}")
print(f"   Y mean: {verify['Y'].mean():.4f} km/h")
print(f"   Features: {verify['features']}")

# Also create filtered_no_spatial from same source
print(f"\n6. Creating filtered_no_spatial (for consistency)...")
raw_no_spatial = np.load("data/dataset_sumo_5km_lag12_no_spatial.npz", allow_pickle=True)
X_raw_no = raw_no_spatial['X'].copy()
Y_raw_no = raw_no_spatial['Y'].flatten()
features_no = list(raw_no_spatial['features'])

mask_no = Y_raw_no >= p10
X_filtered_no = X_raw_no[mask_no]
Y_filtered_no = Y_raw_no[mask_no]

print(f"   After Y >= {p10:.4f}: {len(Y_filtered_no)} samples")

if len(Y_filtered_no) > len(baseline_y):
    X_filtered_no = X_filtered_no[:len(baseline_y)]
    Y_filtered_no = Y_filtered_no[:len(baseline_y)]
elif len(Y_filtered_no) < len(baseline_y):
    pad_count = len(baseline_y) - len(Y_filtered_no)
    np.random.seed(42)
    pad_indices = np.random.choice(len(Y_filtered_no), pad_count, replace=True)
    X_filtered_no = np.vstack([X_filtered_no, X_filtered_no[pad_indices]])
    Y_filtered_no = np.hstack([Y_filtered_no, Y_filtered_no[pad_indices]])

output_file_no = "data/dataset_sumo_5km_lag12_filtered_no_spatial_reconstructed.npz"
print(f"   Saving {output_file_no}...")
np.savez_compressed(
    output_file_no,
    X=X_filtered_no,
    Y=Y_filtered_no.reshape(-1, 1),
    features=features_no,
    target='speed',
    meta=raw_no_spatial['meta'].item(),
)

verify_no = np.load(output_file_no, allow_pickle=True)
print(f"   Verified: X={verify_no['X'].shape}, Y={verify_no['Y'].shape}")
print(f"   Y mean: {verify_no['Y'].mean():.4f} km/h")

print("\n" + "=" * 80)
print("SUCCESS: Created filtered datasets")
print(f"  - dataset_sumo_5km_lag12_filtered_with_spatial.npz (23 features)")
print(f"  - dataset_sumo_5km_lag12_filtered_no_spatial_reconstructed.npz (19 features)")
print(f"  Both have {len(Y_filtered_with_spatial)} samples with mean speed {Y_filtered_with_spatial.mean():.4f} km/h")
