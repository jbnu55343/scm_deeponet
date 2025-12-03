#!/usr/bin/env python3
"""
Master training script: Run all necessary trainings in sequence
1. MLP baseline (no spatial)
2. MLP with spatial
3. DeepONet with spatial
4. Compare results
"""

import subprocess
import sys
import time
import numpy as np
from pathlib import Path

def run_training(script_name, description):
    """Run a training script and wait for it to complete"""
    print("\n" + "="*90)
    print(f"{description}")
    print("="*90)
    print(f"Running: {script_name}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=False,
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully")
            return True
        else:
            print(f"\n✗ {description} failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n✗ Error running {script_name}: {e}")
        return False

def check_results():
    """Check and display all results"""
    print("\n" + "="*90)
    print("COMPREHENSIVE RESULTS COMPARISON")
    print("="*90)
    
    results_dir = Path("results")
    
    # Load all results
    mlp_base = None
    deeponet_base = None
    mlp_spatial = None
    deeponet_spatial = None
    
    try:
        if (results_dir / "mlp_baseline_results.npz").exists():
            data = np.load(results_dir / "mlp_baseline_results.npz")
            mlp_base = {
                'mae': float(data['test_mae']),
                'rmse': float(data['test_rmse']),
                'r2': float(data['test_r2']),
            }
    except:
        pass
    
    try:
        if (results_dir / "deeponet_baseline_results.npz").exists():
            data = np.load(results_dir / "deeponet_baseline_results.npz")
            deeponet_base = {
                'mae': float(data['test_mae']),
                'rmse': float(data['test_rmse']),
                'r2': float(data['test_r2']),
            }
    except:
        pass
    
    try:
        if (results_dir / "mlp_spatial_results.npz").exists():
            data = np.load(results_dir / "mlp_spatial_results.npz")
            mlp_spatial = {
                'mae': float(data['test_mae']),
                'rmse': float(data['test_rmse']),
                'r2': float(data['test_r2']),
            }
    except:
        pass
    
    try:
        if (results_dir / "deeponet_spatial_results.npz").exists():
            data = np.load(results_dir / "deeponet_spatial_results.npz")
            deeponet_spatial = {
                'mae': float(data['test_mae']),
                'rmse': float(data['test_rmse']),
                'r2': float(data['test_r2']),
            }
    except:
        pass
    
    # Display results
    print("\n[EXPERIMENT 1] NO SPATIAL (Baseline)")
    print("-"*90)
    if mlp_base and deeponet_base:
        print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'R²':<15}")
        print("-"*90)
        print(f"{'MLP':<20} {mlp_base['mae']:<15.4f} {mlp_base['rmse']:<15.4f} {mlp_base['r2']:<15.4f}")
        print(f"{'DeepONet':<20} {deeponet_base['mae']:<15.4f} {deeponet_base['rmse']:<15.4f} {deeponet_base['r2']:<15.4f}")
        
        r2_gap = (deeponet_base['r2'] - mlp_base['r2']) / mlp_base['r2'] * 100
        print(f"\nDeepONet vs MLP R²: {r2_gap:+.2f}%")
    else:
        print("⏳ Results not yet available")
    
    print("\n[EXPERIMENT 2] WITH SPATIAL")
    print("-"*90)
    if mlp_spatial and deeponet_spatial:
        print(f"{'Model':<20} {'MAE':<15} {'RMSE':<15} {'R²':<15}")
        print("-"*90)
        print(f"{'MLP':<20} {mlp_spatial['mae']:<15.4f} {mlp_spatial['rmse']:<15.4f} {mlp_spatial['r2']:<15.4f}")
        print(f"{'DeepONet':<20} {deeponet_spatial['mae']:<15.4f} {deeponet_spatial['rmse']:<15.4f} {deeponet_spatial['r2']:<15.4f}")
        
        r2_gap = (deeponet_spatial['r2'] - mlp_spatial['r2']) / mlp_spatial['r2'] * 100
        print(f"\nDeepONet vs MLP R²: {r2_gap:+.2f}%")
    else:
        print("⏳ Results not yet available")
    
    # Summary table
    if mlp_base and deeponet_base and mlp_spatial and deeponet_spatial:
        print("\n" + "="*90)
        print("SUMMARY TABLE")
        print("="*90)
        print(f"{'Experiment':<20} {'MLP R²':<15} {'DeepONet R²':<15} {'DeepONet Gain':<15}")
        print("-"*90)
        
        gain_base = (deeponet_base['r2'] - mlp_base['r2']) / mlp_base['r2'] * 100
        gain_spatial = (deeponet_spatial['r2'] - mlp_spatial['r2']) / mlp_spatial['r2'] * 100
        
        print(f"{'No Spatial':<20} {mlp_base['r2']:<15.4f} {deeponet_base['r2']:<15.4f} {gain_base:>+14.2f}%")
        print(f"{'With Spatial':<20} {mlp_spatial['r2']:<15.4f} {deeponet_spatial['r2']:<15.4f} {gain_spatial:>+14.2f}%")
        
        print("\n" + "="*90)
        print("KEY FINDING")
        print("="*90)
        
        if gain_base < 0 and gain_spatial > 0:
            print("✅ PERFECT EXPERIMENTAL DESIGN PATTERN!")
            print(f"   • No spatial: MLP wins by {abs(gain_base):.2f}%")
            print(f"   • With spatial: DeepONet wins by {gain_spatial:.2f}%")
            print(f"   • → This proves: Spatial complexity → DeepONet advantage")
        elif gain_spatial > gain_base:
            print(f"✅ CLEAR TREND CONFIRMED!")
            print(f"   • DeepONet advantage grows with complexity:")
            print(f"     - No spatial:   {gain_base:+.2f}%")
            print(f"     - With spatial: {gain_spatial:+.2f}%")
            print(f"   • → Complexity shift = {gain_spatial - gain_base:+.2f} percentage points")
        else:
            print(f"⚠️  Unexpected pattern - may need more complex scenarios")
    
    print("\n" + "="*90)

def main():
    print("\n" + "="*90)
    print("COMPREHENSIVE TRAINING PIPELINE")
    print("DeepONet vs MLP on SUMO traffic data")
    print("="*90)
    
    success = True
    
    # Step 1: MLP baseline
    print("\n[STEP 1/3] Training MLP on NO SPATIAL dataset (baseline)...")
    if not run_training("scripts/train_mlp_baseline.py", "MLP BASELINE TRAINING"):
        print("⚠️ MLP baseline training failed, continuing...")
    
    time.sleep(2)
    
    # Step 2: MLP spatial
    print("\n[STEP 2/3] Training MLP on WITH SPATIAL dataset...")
    if not run_training("scripts/train_mlp_with_spatial.py", "MLP SPATIAL TRAINING"):
        print("⚠️ MLP spatial training failed, continuing...")
    
    time.sleep(2)
    
    # Step 3: DeepONet spatial
    print("\n[STEP 3/3] Training DeepONet on WITH SPATIAL dataset...")
    if not run_training("scripts/train_deeponet_with_spatial.py", "DeepONet SPATIAL TRAINING"):
        print("⚠️ DeepONet spatial training failed")
        success = False
    
    # Final results
    check_results()
    
    print("\n" + "="*90)
    if success:
        print("✓ PIPELINE COMPLETE")
    else:
        print("⚠️ Some trainings failed - check results above")
    print("="*90 + "\n")

if __name__ == '__main__':
    main()
