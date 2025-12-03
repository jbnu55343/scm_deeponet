#!/usr/bin/env python3
"""
Quick diagnostics and re-train MLP baseline if needed
"""

import os
import numpy as np
from pathlib import Path

# Check what we have
print("\n" + "="*80)
print("CURRENT TRAINING STATUS")
print("="*80)

results_dir = Path("results")
models_dir = Path("models")

# Check baseline results
mlp_baseline_file = results_dir / "mlp_baseline_results.npz"
deeponet_baseline_file = results_dir / "deeponet_baseline_results.npz"

print("\n[Baseline Results]")
if mlp_baseline_file.exists():
    try:
        data = np.load(mlp_baseline_file)
        print(f"✓ MLP baseline: R²={float(data['test_r2']):.4f}")
    except:
        print(f"✗ MLP baseline exists but can't read")
else:
    print(f"✗ MLP baseline MISSING")

if deeponet_baseline_file.exists():
    try:
        data = np.load(deeponet_baseline_file)
        print(f"✓ DeepONet baseline: R²={float(data['test_r2']):.4f}")
    except:
        print(f"✗ DeepONet baseline exists but can't read")
else:
    print(f"✗ DeepONet baseline MISSING")

# Check spatial results
mlp_spatial_file = results_dir / "mlp_spatial_results.npz"
deeponet_spatial_file = results_dir / "deeponet_spatial_results.npz"

print("\n[Spatial Results]")
if mlp_spatial_file.exists():
    try:
        data = np.load(mlp_spatial_file)
        print(f"✓ MLP spatial: R²={float(data['test_r2']):.4f}")
    except:
        print(f"✗ MLP spatial exists but can't read")
else:
    print(f"✗ MLP spatial MISSING")

if deeponet_spatial_file.exists():
    try:
        data = np.load(deeponet_spatial_file)
        print(f"✓ DeepONet spatial: R²={float(data['test_r2']):.4f}")
    except:
        print(f"✗ DeepONet spatial exists but can't read")
else:
    print(f"✗ DeepONet spatial MISSING")

# Check model files
print("\n[Model Files]")
mlp_baseline_model = models_dir / "mlp_baseline.pt"
deeponet_baseline_model = models_dir / "deeponet_baseline.pt"
mlp_spatial_model = models_dir / "mlp_spatial.pt"
deeponet_spatial_model = models_dir / "deeponet_spatial.pt"

print(f"{'MLP baseline:':<20} {'✓' if mlp_baseline_model.exists() else '✗'}")
print(f"{'DeepONet baseline:':<20} {'✓' if deeponet_baseline_model.exists() else '✗'}")
print(f"{'MLP spatial:':<20} {'✓' if mlp_spatial_model.exists() else '✗'}")
print(f"{'DeepONet spatial:':<20} {'✓' if deeponet_spatial_model.exists() else '✗'}")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

missing = []
if not mlp_baseline_file.exists():
    missing.append("MLP baseline results")
if not mlp_spatial_file.exists():
    missing.append("MLP spatial results")
if not deeponet_spatial_file.exists():
    missing.append("DeepONet spatial results")

if missing:
    print("\nMissing training outputs:")
    for item in missing:
        print(f"  - {item}")
    print("\nRun these commands in order:")
    print(f"  1. python scripts/train_mlp_baseline.py    (if missing MLP baseline)")
    print(f"  2. python scripts/train_mlp_with_spatial.py")
    print(f"  3. python scripts/train_deeponet_with_spatial.py")
    print(f"\nOr run: python scripts/train_all_spatial.py (to automate everything)")
else:
    print("\n✓ All required training outputs exist!")
    print("\nNow run: python scripts/comprehensive_comparison.py")
