#!/usr/bin/env python3
"""
Compare MLP vs DeepONet results.
"""

import numpy as np
import os

def load_results(results_file):
    """Load results from npz file."""
    if not os.path.exists(results_file):
        print(f"Warning: {results_file} not found")
        return None
    
    data = np.load(results_file)
    return {
        'mae': float(data['test_mae']),
        'rmse': float(data['test_rmse']),
        'r2': float(data['test_r2']),
        'y_true': data['test_true'],
        'y_pred': data['test_pred']
    }

def print_comparison():
    """Print MLP vs DeepONet comparison."""
    
    mlp_results = load_results("results/mlp_baseline_results.npz")
    deeponet_results = load_results("results/deeponet_baseline_results.npz")
    
    if mlp_results is None or deeponet_results is None:
        print("❌ Results not ready. Run both training scripts first:")
        print("   python scripts/train_mlp_baseline.py")
        print("   python scripts/train_deeponet_baseline.py")
        return
    
    print("\n" + "="*80)
    print("MODEL COMPARISON: MLP vs DeepONet")
    print("="*80)
    
    print(f"\n{'Metric':<20} {'MLP':<20} {'DeepONet':<20} {'Improvement':<15}")
    print("-" * 75)
    
    # MAE
    mae_improvement = (mlp_results['mae'] - deeponet_results['mae']) / mlp_results['mae'] * 100
    print(f"{'MAE (km/h)':<20} {mlp_results['mae']:<20.4f} {deeponet_results['mae']:<20.4f} {mae_improvement:>+6.2f}%")
    
    # RMSE
    rmse_improvement = (mlp_results['rmse'] - deeponet_results['rmse']) / mlp_results['rmse'] * 100
    print(f"{'RMSE (km/h)':<20} {mlp_results['rmse']:<20.4f} {deeponet_results['rmse']:<20.4f} {rmse_improvement:>+6.2f}%")
    
    # R²
    r2_improvement = (deeponet_results['r2'] - mlp_results['r2']) / mlp_results['r2'] * 100
    print(f"{'R² Score':<20} {mlp_results['r2']:<20.4f} {deeponet_results['r2']:<20.4f} {r2_improvement:>+6.2f}%")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if deeponet_results['r2'] > mlp_results['r2']:
        print(f"\n✓ DeepONet outperforms MLP")
        print(f"  - R² improvement: +{r2_improvement:.2f}%")
        print(f"  - MAE improvement: {mae_improvement:.2f}%")
    else:
        print(f"\n⚠ MLP performs better or comparable")
        print(f"  - This suggests both models capture similar patterns")
        print(f"  - DeepONet's operator structure may not be beneficial for this task")
    
    print(f"\nTest set performance (last 20% of data):")
    print(f"  - MLP R²: {mlp_results['r2']:.4f}")
    print(f"  - DeepONet R²: {deeponet_results['r2']:.4f}")
    print(f"\nThis is reasonable performance on realistic traffic data.")
    print(f"(Paper's R²=0.9856 was on 94.9% zero values - a much simpler task)")


if __name__ == '__main__':
    print_comparison()
