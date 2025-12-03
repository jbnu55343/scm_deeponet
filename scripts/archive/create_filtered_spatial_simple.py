#!/usr/bin/env python3
"""
Create filtered dataset with spatial features
by combining filtered baseline + spatial features from raw with_spatial
"""

import numpy as np

print("Creating Filtered Dataset WITH Spatial Features")
print("=" * 80)

# Load filtered baseline (no spatial) - this is our target template
print("\nLoading filtered baseline (no spatial)...")
filtered_base = np.load("data/dataset_sumo_5km_lag12_filtered.npz", allow_pickle=True)
X_filtered_base = filtered_base['X'].copy()
Y_filtered_base = filtered_base['Y'].copy()
meta_filtered = filtered_base['meta'].item()
print(f"  Filtered baseline: X={X_filtered_base.shape}, Y={Y_filtered_base.shape}")
print(f"  Y mean: {Y_filtered_base.mean():.4f} km/h")
print(f"  Y zeros: {(Y_filtered_base == 0).sum()}")

# Load raw with_spatial to get the 4 spatial features
print("\nLoading raw data with spatial features...")
raw_spatial = np.load("data/dataset_sumo_5km_lag12_with_spatial.npz", allow_pickle=True)
X_raw_spatial = raw_spatial['X'].copy()
Y_raw_spatial = raw_spatial['Y'].copy()
features_raw_spatial = list(raw_spatial['features'])
print(f"  Raw with_spatial: X={X_raw_spatial.shape}, Y={Y_raw_spatial.shape}")
print(f"  Features: {len(features_raw_spatial)}")

# The spatial features are inserted at indices 7-10 in with_spatial
# Positions: [7] speed_upstream_mean, [8] speed_downstream_mean, [9] density_upstream_mean, [10] density_downstream_mean
spatial_feature_indices = [7, 8, 9, 10]
print(f"\nSpatial feature indices: {spatial_feature_indices}")
print(f"  Feature names: {[features_raw_spatial[i] for i in spatial_feature_indices]}")

# Filter raw_spatial to get the same samples as filtered_base
# Criterion: Y > 0
mask_spatial = Y_raw_spatial.reshape(-1) > 0
Y_raw_filtered = Y_raw_spatial[mask_spatial]
X_raw_filtered = X_raw_spatial[mask_spatial]

print(f"\nRaw data after Y > 0 filtering: {len(Y_raw_filtered)} samples")
print(f"Filtered base has: {len(Y_filtered_base)} samples")

if len(Y_raw_filtered) != len(Y_filtered_base):
    print(f"WARNING: Sample count mismatch!")
    print(f"  Difference: {len(Y_raw_filtered) - len(Y_filtered_base)}")
    print(f"  Raw is larger, using first {len(Y_filtered_base)} samples after filtering")
    # Use only the first N samples to match filtered_base
    X_raw_filtered = X_raw_filtered[:len(Y_filtered_base)]
    Y_raw_filtered = Y_raw_filtered[:len(Y_filtered_base)]

# Extract spatial features
X_spatial_features = X_raw_filtered[:, spatial_feature_indices]
print(f"\nExtracted spatial features: {X_spatial_features.shape}")
print(f"  Spatial features mean: {X_spatial_features.mean():.4f}")
print(f"  Spatial features zeros: {(X_spatial_features == 0).sum()}")

# Combine: filtered baseline (19 features) + spatial (4 features) = 23 features
X_combined = np.hstack([X_filtered_base, X_spatial_features])
Y_combined = Y_filtered_base  # Y should be identical

print(f"\nCombined dataset: X={X_combined.shape}, Y={Y_combined.shape}")
print(f"  Y mean: {Y_combined.mean():.4f} km/h")
print(f"  Y zeros: {(Y_combined == 0).sum()}")

# Update features list
features_no_spatial = list(filtered_base['features'])
features_spatial = features_raw_spatial[-4:]
features_combined = features_no_spatial + features_spatial

print(f"\nFeature list ({len(features_combined)}):")
for i, f in enumerate(features_combined[-5:]):
    print(f"  [{i + len(features_combined) - 5}] {f}")

# Create metadata
meta_combined = meta_filtered.copy()
meta_combined['features'] = features_combined
meta_combined['n_spatial_features'] = 4

# Save
output_file = "data/dataset_sumo_5km_lag12_filtered_with_spatial.npz"
print(f"\nSaving to {output_file}...")
np.savez_compressed(
    output_file,
    X=X_combined,
    Y=Y_combined,
    features=features_combined,
    target='speed',
    meta=meta_combined,
)

# Verify
print("\nVerifying saved file...")
verify = np.load(output_file, allow_pickle=True)
print(f"  X shape: {verify['X'].shape}")
print(f"  Y shape: {verify['Y'].shape}")
print(f"  Features: {len(verify['features'])}")
print(f"  Y mean: {verify['Y'].mean():.4f} km/h")

print("\n" + "=" * 80)
print("SUCCESS: Created dataset_sumo_5km_lag12_filtered_with_spatial.npz")
print(f"  Samples: {len(Y_combined)}")
print(f"  Features: {X_combined.shape[1]}")
print(f"  Mean speed: {Y_combined.mean():.4f} km/h")
