#!/usr/bin/env python3
"""
Generate baseline comparison with known results from conversation history
"""

import numpy as np
from pathlib import Path

# Create results directory if not exists
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# From conversation history:
# MLP baseline (no spatial, filtered data, 1.2M samples): R²=0.8103, MAE=2.7462
# DeepONet baseline (same): R²=0.7914, MAE=2.8503

# But we need to check - the spatial dataset is 6.6M samples (unfiltered)
# So we should verify the baseline metrics match or are reasonable

print("\n" + "="*80)
print("BASELINE RESULTS STATUS")
print("="*80)

# Check if we have baseline results
mlp_baseline_file = results_dir / "mlp_baseline_results.npz"
deeponet_baseline_file = results_dir / "deeponet_baseline_results.npz"

if mlp_baseline_file.exists():
    try:
        data = np.load(mlp_baseline_file)
        print(f"\n✓ MLP Baseline found:")
        print(f"  R²={float(data['test_r2']):.4f}")
        print(f"  MAE={float(data['test_mae']):.4f}")
        print(f"  RMSE={float(data['test_rmse']):.4f}")
    except Exception as e:
        print(f"\n✗ Error reading MLP baseline: {e}")
else:
    print(f"\n✗ MLP baseline not found at {mlp_baseline_file}")
    print("  Need to train MLP baseline first")

if deeponet_baseline_file.exists():
    try:
        data = np.load(deeponet_baseline_file)
        print(f"\n✓ DeepONet Baseline found:")
        print(f"  R²={float(data['test_r2']):.4f}")
        print(f"  MAE={float(data['test_mae']):.4f}")
        print(f"  RMSE={float(data['test_rmse']):.4f}")
    except Exception as e:
        print(f"\n✗ Error reading DeepONet baseline: {e}")
else:
    print(f"\n✗ DeepONet baseline not found at {deeponet_baseline_file}")

print("\n" + "="*80)

# Check spatial results
mlp_spatial_file = results_dir / "mlp_spatial_results.npz"
deeponet_spatial_file = results_dir / "deeponet_spatial_results.npz"

print("\nSPATIAL RESULTS STATUS")
print("="*80)

if mlp_spatial_file.exists():
    try:
        data = np.load(mlp_spatial_file)
        print(f"\n✓ MLP Spatial found:")
        print(f"  R²={float(data['test_r2']):.4f}")
        print(f"  MAE={float(data['test_mae']):.4f}")
        print(f"  RMSE={float(data['test_rmse']):.4f}")
    except Exception as e:
        print(f"\n✗ Error reading MLP spatial: {e}")
else:
    print(f"\n✗ MLP spatial not yet available (training in progress...)")

if deeponet_spatial_file.exists():
    try:
        data = np.load(deeponet_spatial_file)
        print(f"\n✓ DeepONet Spatial found:")
        print(f"  R²={float(data['test_r2']):.4f}")
        print(f"  MAE={float(data['test_mae']):.4f}")
        print(f"  RMSE={float(data['test_rmse']):.4f}")
    except Exception as e:
        print(f"\n✗ Error reading DeepONet spatial: {e}")
else:
    print(f"\n✗ DeepONet spatial not yet available (training in progress...)")

print("\n" + "="*80)
