#!/usr/bin/env python3
"""
Quick analysis: How much do spatial features contribute?
Linear regression comparison: temporal only vs temporal+spatial
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import time

print("="*80)
print("SPATIAL FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Load datasets
print("\nLoading data...")
filtered_no_spatial = np.load("data/dataset_sumo_5km_lag12_filtered_no_spatial_reconstructed.npz")
filtered_with_spatial = np.load("data/dataset_sumo_5km_lag12_filtered_with_spatial.npz")

X_temporal = filtered_no_spatial['X'].astype(np.float32)  # (N, 19)
X_temporal_spatial = filtered_with_spatial['X'].astype(np.float32)  # (N, 23)
Y = filtered_with_spatial['Y'].astype(np.float32).flatten()  # (N,)

print(f"X temporal: {X_temporal.shape}")
print(f"X temporal+spatial: {X_temporal_spatial.shape}")
print(f"Y: {Y.shape}, mean={Y.mean():.4f}, std={Y.std():.4f}")

# Extract spatial features
X_spatial_only = X_temporal_spatial[:, 19:23]
print(f"\nSpatial features (4 new): {X_spatial_only.shape}")
print(f"  Spatial mean: {X_spatial_only.mean(axis=0)}")
print(f"  Spatial std:  {X_spatial_only.std(axis=0)}")

# 80/20 split
split_idx = int(0.8 * len(X_temporal))
X_train_temporal = X_temporal[:split_idx]
X_test_temporal = X_temporal[split_idx:]
X_train_spatial = X_temporal_spatial[:split_idx]
X_test_spatial = X_temporal_spatial[split_idx:]
Y_train = Y[:split_idx]
Y_test = Y[split_idx:]

print(f"\nTrain/test split: {len(Y_train)} / {len(Y_test)}")

# ============================================================================
# LINEAR REGRESSION: Temporal Only
# ============================================================================
print("\n" + "="*80)
print("LINEAR REGRESSION: Temporal Features Only (19 features)")
print("="*80)

start = time.time()
lr_temporal = LinearRegression()
lr_temporal.fit(X_train_temporal, Y_train)
elapsed = time.time() - start

Y_pred_temporal = lr_temporal.predict(X_test_temporal)
r2_temporal = r2_score(Y_test, Y_pred_temporal)
mae_temporal = mean_absolute_error(Y_test, Y_pred_temporal)

print(f"Train time: {elapsed:.3f}s")
print(f"Test R²:  {r2_temporal:.6f}")
print(f"Test MAE: {mae_temporal:.6f}")

# ============================================================================
# LINEAR REGRESSION: Temporal + Spatial
# ============================================================================
print("\n" + "="*80)
print("LINEAR REGRESSION: Temporal + Spatial Features (23 features)")
print("="*80)

start = time.time()
lr_spatial = LinearRegression()
lr_spatial.fit(X_train_spatial, Y_train)
elapsed = time.time() - start

Y_pred_spatial = lr_spatial.predict(X_test_spatial)
r2_spatial = r2_score(Y_test, Y_pred_spatial)
mae_spatial = mean_absolute_error(Y_test, Y_pred_spatial)

print(f"Train time: {elapsed:.3f}s")
print(f"Test R²:  {r2_spatial:.6f}")
print(f"Test MAE: {mae_spatial:.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("SPATIAL FEATURE IMPORTANCE")
print("="*80)

r2_delta = r2_spatial - r2_temporal
mae_delta = mae_spatial - mae_temporal
r2_pct_change = (r2_delta / abs(r2_temporal)) * 100 if r2_temporal != 0 else 0
mae_pct_change = (mae_delta / abs(mae_temporal)) * 100 if mae_temporal != 0 else 0

print(f"\nR² Change:   {r2_temporal:.6f} → {r2_spatial:.6f} ({r2_delta:+.6f}, {r2_pct_change:+.2f}%)")
print(f"MAE Change:  {mae_temporal:.6f} → {mae_spatial:.6f} ({mae_delta:+.6f}, {mae_pct_change:+.2f}%)")

if r2_spatial > r2_temporal:
    print(f"\n✓ Spatial features HELP linear model (R² improves by {r2_delta:.6f})")
else:
    print(f"\n✗ Spatial features HURT linear model (R² worsens by {abs(r2_delta):.6f})")
    print("  → This suggests spatial features may be collinear or noisy")

# ============================================================================
# FEATURE IMPORTANCE from Linear Regression Coefficients
# ============================================================================
print("\n" + "="*80)
print("SPATIAL FEATURE COEFFICIENTS (from linear regression)")
print("="*80)

feature_names = list(filtered_with_spatial['features'])
print(f"\nTemporal+Spatial model coefficients (top 5 by magnitude):")

coefs = lr_spatial.coef_
abs_coefs = np.abs(coefs)
top_indices = np.argsort(abs_coefs)[-5:][::-1]

for idx in top_indices:
    fname = feature_names[idx]
    coef = coefs[idx]
    print(f"  [{idx:2d}] {fname:30s}: {coef:+.8f}")

print(f"\nSpatial feature coefficients:")
spatial_names = ['speed_upstream_mean', 'speed_downstream_mean', 'density_upstream_mean', 'density_downstream_mean']
for i, sname in enumerate(spatial_names):
    idx = 19 + i
    coef = coefs[idx]
    print(f"  [{idx}] {sname:30s}: {coef:+.8f}")

# ============================================================================
# FEATURE CORRELATION WITH TARGET
# ============================================================================
print("\n" + "="*80)
print("SPATIAL FEATURE CORRELATIONS WITH TARGET")
print("="*80)

print(f"\nCorrelation of each spatial feature with Y:")
for i, sname in enumerate(spatial_names):
    col = X_temporal_spatial[:, 19 + i]
    corr = np.corrcoef(col, Y)[0, 1]
    print(f"  {sname:30s}: {corr:+.6f}")

print(f"\nCorrelation of temporal features with Y (top 3):")
temporal_corrs = []
for i in range(19):
    col = X_temporal[:, i]
    corr = np.corrcoef(col, Y)[0, 1]
    temporal_corrs.append((i, feature_names[i], corr))

temporal_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
for i, fname, corr in temporal_corrs[:3]:
    print(f"  {fname:30s}: {corr:+.6f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

spatial_corr_strengths = []
for i, sname in enumerate(spatial_names):
    col = X_temporal_spatial[:, 19 + i]
    corr = abs(np.corrcoef(col, Y)[0, 1])
    spatial_corr_strengths.append(corr)

print(f"\nAverage spatial feature correlation: {np.mean(spatial_corr_strengths):.6f}")
print(f"Average temporal feature correlation: {np.mean([abs(c) for _, _, c in temporal_corrs]):.6f}")

if r2_spatial > r2_temporal:
    print(f"\n✓ SUCCESS: Spatial features improve linear model by {r2_pct_change:.2f}%")
    print("  → Ready for neural network experiments")
elif abs(r2_delta) < 0.001:
    print(f"\n~ NEUTRAL: Spatial features have minimal linear impact ({r2_pct_change:.2f}%)")
    print("  → DeepONet may still benefit (non-linear interactions)")
else:
    print(f"\n✗ WARNING: Spatial features hurt linear model by {abs(r2_pct_change):.2f}%")
    print("  Possible causes:")
    print("    1. Feature scaling (different ranges)")
    print("    2. Multicollinearity (correlated with temporal features)")
    print("    3. Data quality (sparse, zero-heavy)")
    print("\n  But DeepONet might still use them via non-linear combinations")
