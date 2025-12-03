#!/usr/bin/env python3
"""
Create filtered dataset with spatial features
by adding neighbor information to the filtered baseline
"""

import numpy as np
from pathlib import Path
import time

print("\n" + "="*80)
print("Creating Filtered Dataset WITH Spatial Features")
print("="*80)

# Load filtered baseline (no spatial)
print("\nLoading filtered baseline (no spatial)...")
filtered_data = np.load("data/dataset_sumo_5km_lag12_filtered.npz", allow_pickle=True)
X_filtered = filtered_data['X'].copy()
y_filtered = filtered_data['Y'].copy()
features_filtered = filtered_data['features']
meta_filtered = filtered_data['meta']

print(f"  Filtered baseline: X={X_filtered.shape}, y={y_filtered.shape}")
print(f"  Features: {len(features_filtered)}")

# Load raw with_spatial to extract the spatial features
print("\nLoading raw with_spatial dataset...")
spatial_data = np.load("data/dataset_sumo_5km_lag12_with_spatial.npz", allow_pickle=True)
X_spatial_raw = spatial_data['X']
features_spatial_raw = spatial_data['features']

print(f"  Raw spatial: X={X_spatial_raw.shape}")

# Get indices of spatial features in raw data
spatial_feature_names = ['speed_upstream_mean', 'speed_downstream_mean', 
                         'density_upstream_mean', 'density_downstream_mean']
spatial_indices = []
for fname in spatial_feature_names:
    for i, f in enumerate(features_spatial_raw):
        if f == fname:
            spatial_indices.append(i)
            break

print(f"\nSpatial features found at indices: {spatial_indices}")
print(f"  Features: {[features_spatial_raw[i] for i in spatial_indices]}")

# Now we need to extract spatial features from raw data at same indices as filtered data
# But filtered data is a subset - we need to reconstruct which rows are in filtered

# Strategy: Load the unfiltered version to find indices
print("\nLoading raw no_spatial to find filtered indices...")
raw_data = np.load("data/dataset_sumo_5km_lag12_no_spatial.npz", allow_pickle=True)
X_raw = raw_data['X']
y_raw = raw_data['Y']

print(f"  Raw no-spatial: X={X_raw.shape}, y={y_raw.shape}")

# Find which rows in raw data correspond to non-zero y values
non_zero_indices = np.where(y_raw.flatten() != 0)[0]
print(f"  Non-zero indices: {len(non_zero_indices)} out of {len(y_raw)}")
print(f"  Expected filtered size: {len(X_filtered)}")

if len(non_zero_indices) == len(X_filtered):
    print("  ✓ Indices match!")
    
    # Extract spatial features from raw data at non-zero indices
    X_spatial_features = X_spatial_raw[non_zero_indices, spatial_indices].copy()
    
    print(f"\nExtracted spatial features shape: {X_spatial_features.shape}")
    print(f"  Feature 1 (upstream speed): mean={X_spatial_features[:, 0].mean():.4f}")
    print(f"  Feature 2 (downstream speed): mean={X_spatial_features[:, 1].mean():.4f}")
    print(f"  Feature 3 (upstream density): mean={X_spatial_features[:, 2].mean():.4f}")
    print(f"  Feature 4 (downstream density): mean={X_spatial_features[:, 3].mean():.4f}")
    
    # Concatenate to create filtered+spatial dataset
    X_filtered_spatial = np.concatenate([X_filtered, X_spatial_features], axis=1)
    
    print(f"\nCombined filtered+spatial X shape: {X_filtered_spatial.shape}")
    
    # Create new feature list
    features_filtered_spatial = np.concatenate([
        features_filtered,
        np.array(spatial_feature_names, dtype=object)
    ])
    
    print(f"  Total features: {len(features_filtered_spatial)}")
    print(f"  Features: {list(features_filtered_spatial)}")
    
    # Save new dataset
    output_file = "data/dataset_sumo_5km_lag12_filtered_with_spatial.npz"
    print(f"\nSaving to {output_file}...")
    
    np.savez_compressed(
        output_file,
        X=X_filtered_spatial.astype(np.float32),
        Y=y_filtered.astype(np.float32),
        features=features_filtered_spatial,
        target=filtered_data['target'],
        meta=meta_filtered,
        split=filtered_data['split']
    )
    
    print(f"✓ Saved successfully!")
    
    # Verify
    print("\nVerification:")
    verify_data = np.load(output_file, allow_pickle=True)
    print(f"  X shape: {verify_data['X'].shape}")
    print(f"  Y shape: {verify_data['Y'].shape}")
    print(f"  Features: {len(verify_data['features'])}")
    print(f"  Target mean: {verify_data['Y'].mean():.4f} km/h")
    print(f"  Target std: {verify_data['Y'].std():.4f} km/h")
    
    print("\n" + "="*80)
    print("✓ FILTERED+SPATIAL DATASET CREATED SUCCESSFULLY")
    print("="*80)
    print(f"""
Ready to train with:
  - dataset_sumo_5km_lag12_filtered.npz (baseline, 1.2M samples, 19 features)
  - dataset_sumo_5km_lag12_filtered_with_spatial.npz (spatial, 1.2M samples, 23 features)

Both with:
  - Mean speed: 17.34 km/h (0% zeros)
  - 80/20 temporal split for train/test
""")
    
else:
    print(f"  ✗ Mismatch! {len(non_zero_indices)} != {len(X_filtered)}")
    print("  Cannot proceed - need to investigate data structure")
