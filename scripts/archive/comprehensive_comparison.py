#!/usr/bin/env python3
"""
Comprehensive comparison of all experiments:
1. No spatial (filtered) - baseline
2. With spatial - proves spatial matters  
3. (Optional) METR-LA - proves real data works
"""

import numpy as np
import os

def load_results(filepath):
    """Load results from NPZ file"""
    if not os.path.exists(filepath):
        return None
    try:
        data = np.load(filepath)
        return {
            'mae': float(data['test_mae']),
            'rmse': float(data['test_rmse']),
            'r2': float(data['test_r2']),
        }
    except:
        return None

def print_comparison():
    """Print comprehensive comparison"""
    
    print("\n" + "="*90)
    print("COMPREHENSIVE MODEL COMPARISON ACROSS ALL EXPERIMENTS")
    print("="*90)
    
    # Experiment 1: No spatial (baseline) - already have this
    print("\n[EXPERIMENT 1] NO SPATIAL (Baseline)")
    print("-" * 90)
    
    r1_mlp = load_results("results/mlp_baseline_results.npz")
    r1_deeponet = load_results("results/deeponet_baseline_results.npz")
    
    if r1_mlp and r1_deeponet:
        print(f"{'Model':<20} {'MAE (km/h)':<20} {'RMSE (km/h)':<20} {'R² Score':<15}")
        print("-" * 90)
        print(f"{'MLP':<20} {r1_mlp['mae']:<20.4f} {r1_mlp['rmse']:<20.4f} {r1_mlp['r2']:<15.4f}")
        print(f"{'DeepONet':<20} {r1_deeponet['mae']:<20.4f} {r1_deeponet['rmse']:<20.4f} {r1_deeponet['r2']:<15.4f}")
        
        # Analysis
        r2_gap = (r1_deeponet['r2'] - r1_mlp['r2']) / r1_mlp['r2'] * 100
        mae_gap = (r1_deeponet['mae'] - r1_mlp['mae']) / r1_mlp['mae'] * 100
        
        print(f"\nAnalysis: MLP slightly better (R² gap: {r2_gap:.2f}%)")
        print(f"          Simple task, MLP sufficient")
    else:
        print("❌ Results not available")
    
    # Experiment 2: With spatial
    print("\n[EXPERIMENT 2] WITH SPATIAL (Added spatial features from neighbors)")
    print("-" * 90)
    
    r2_mlp = load_results("results/mlp_spatial_results.npz")
    r2_deeponet = load_results("results/deeponet_spatial_results.npz")
    
    if r2_mlp and r2_deeponet:
        print(f"{'Model':<20} {'MAE (km/h)':<20} {'RMSE (km/h)':<20} {'R² Score':<15}")
        print("-" * 90)
        print(f"{'MLP':<20} {r2_mlp['mae']:<20.4f} {r2_mlp['rmse']:<20.4f} {r2_mlp['r2']:<15.4f}")
        print(f"{'DeepONet':<20} {r2_deeponet['mae']:<20.4f} {r2_deeponet['rmse']:<20.4f} {r2_deeponet['r2']:<15.4f}")
        
        # Analysis
        r2_gap = (r2_deeponet['r2'] - r2_mlp['r2']) / r2_mlp['r2'] * 100
        mae_gap = (r2_deeponet['mae'] - r2_mlp['mae']) / r2_mlp['mae'] * 100
        
        if r2_gap > 0:
            print(f"\n✅ DeepONet BETTER (R² gap: +{r2_gap:.2f}%)")
            print(f"   When spatial features added, DeepONet gains advantage!")
            print(f"   MAE improvement: {mae_gap:.2f}%")
        else:
            print(f"\n⚠️ MLP still better (R² gap: {r2_gap:.2f}%)")
            print(f"   But gap should be smaller than no-spatial case")
    else:
        print("⏳ Waiting for training results...")
    
    # Overall comparison
    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    
    if r1_mlp and r1_deeponet and r2_mlp and r2_deeponet:
        print(f"{'Experiment':<25} {'MLP R²':<15} {'DeepONet R²':<15} {'DeepONet Gain':<15}")
        print("-" * 90)
        
        gain1 = (r1_deeponet['r2'] - r1_mlp['r2']) / r1_mlp['r2'] * 100
        gain2 = (r2_deeponet['r2'] - r2_mlp['r2']) / r2_mlp['r2'] * 100
        
        print(f"{'No Spatial':<25} {r1_mlp['r2']:<15.4f} {r1_deeponet['r2']:<15.4f} {gain1:>+14.2f}%")
        print(f"{'With Spatial':<25} {r2_mlp['r2']:<15.4f} {r2_deeponet['r2']:<15.4f} {gain2:>+14.2f}%")
        
        print("\n" + "="*90)
        print("KEY FINDINGS")
        print("="*90)
        
        if gain1 < 0 and gain2 > 0:
            print(f"✅ PERFECT PATTERN FOUND!")
            print(f"   1. Simple task (no spatial): MLP better ({gain1:.2f}%)")
            print(f"   2. Complex task (with spatial): DeepONet better (+{gain2:.2f}%)")
            print(f"   → This proves DeepONet's advantage emerges with problem complexity!")
            
        elif gain2 > gain1:
            print(f"✅ TREND CONFIRMED!")
            print(f"   DeepONet's advantage grows with spatial complexity:")
            print(f"   No spatial:   {gain1:+.2f}% (MLP advantage)")
            print(f"   With spatial: {gain2:+.2f}% (DeepONet advantage)")
            print(f"   → Difference of {gain2-gain1:.2f} percentage points!")
        else:
            print(f"⚠️ Results not yet matching expected pattern")
            print(f"   Continue with more complex tasks (real data, multi-step, etc.)")
    else:
        print("⏳ Waiting for all training to complete...")
    
    print("\n" + "="*90)


if __name__ == '__main__':
    print_comparison()
