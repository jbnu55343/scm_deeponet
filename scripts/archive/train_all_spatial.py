#!/usr/bin/env python3
"""
Master script: Train both MLP and DeepONet with spatial features
and compare results with baselines
"""

import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

def check_result_file(filepath):
    """Check if result file exists and has metrics"""
    if not os.path.exists(filepath):
        return None
    try:
        data = np.load(filepath)
        return {
            'mae': float(data['test_mae']),
            'rmse': float(data['test_rmse']),
            'r2': float(data['test_r2']),
            'exists': True
        }
    except:
        return None

def wait_for_training(script_name, max_wait=600):
    """Wait for training to complete"""
    print(f"\n⏳ Waiting for {script_name} to complete...")
    print(f"   (checking every 30 seconds, max wait: {max_wait}s)")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        time.sleep(30)
        elapsed = int(time.time() - start_time)
        print(f"   [{elapsed}s elapsed...]", end='\r')
    
    print(f"\n✓ Training period complete")

def main():
    print("\n" + "="*90)
    print("SPATIAL EXPERIMENTS MASTER SCRIPT")
    print("="*90)
    
    # Paths
    mlp_spatial_script = "scripts/train_mlp_with_spatial.py"
    deeponet_spatial_script = "scripts/train_deeponet_with_spatial.py"
    
    mlp_spatial_results = "results/mlp_spatial_results.npz"
    deeponet_spatial_results = "results/deeponet_spatial_results.npz"
    
    mlp_baseline_results = "results/mlp_baseline_results.npz"
    deeponet_baseline_results = "results/deeponet_baseline_results.npz"
    
    print("\n[STEP 1] Launching MLP with spatial training...")
    print(f"  Script: {mlp_spatial_script}")
    print(f"  Will save to: {mlp_spatial_results}")
    
    try:
        proc_mlp = subprocess.Popen(
            ["D:\\DL\\envs\\pytorch_gpu\\python.exe", mlp_spatial_script],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("  ✓ MLP training launched")
    except Exception as e:
        print(f"  ✗ Failed to launch MLP training: {e}")
        return
    
    # Wait for MLP to be mostly done (85 epochs ~ 30-40 min)
    wait_for_training("MLP spatial", max_wait=2400)  # 40 minutes
    
    print("\n[STEP 2] Launching DeepONet with spatial training...")
    print(f"  Script: {deeponet_spatial_script}")
    print(f"  Will save to: {deeponet_spatial_results}")
    
    try:
        proc_deeponet = subprocess.Popen(
            ["D:\\DL\\envs\\pytorch_gpu\\python.exe", deeponet_spatial_script],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("  ✓ DeepONet training launched")
    except Exception as e:
        print(f"  ✗ Failed to launch DeepONet training: {e}")
        return
    
    # Wait for DeepONet
    wait_for_training("DeepONet spatial", max_wait=2400)  # 40 minutes
    
    # Wait a bit more to ensure files are written
    print("\n[STEP 3] Finalizing training...")
    time.sleep(10)
    
    # Load all results
    print("\n[STEP 4] Loading and comparing results...")
    print("="*90)
    
    mlp_base = check_result_file(mlp_baseline_results)
    deeponet_base = check_result_file(deeponet_baseline_results)
    mlp_spatial = check_result_file(mlp_spatial_results)
    deeponet_spatial = check_result_file(deeponet_spatial_results)
    
    print("\n[EXPERIMENT 1] NO SPATIAL (Baseline)")
    print("-"*90)
    if mlp_base and deeponet_base:
        print(f"{'Model':<15} {'MAE':<15} {'RMSE':<15} {'R²':<15}")
        print("-"*90)
        print(f"{'MLP':<15} {mlp_base['mae']:<15.4f} {mlp_base['rmse']:<15.4f} {mlp_base['r2']:<15.4f}")
        print(f"{'DeepONet':<15} {deeponet_base['mae']:<15.4f} {deeponet_base['rmse']:<15.4f} {deeponet_base['r2']:<15.4f}")
        
        r2_gap = (deeponet_base['r2'] - mlp_base['r2']) / mlp_base['r2'] * 100
        print(f"\nDeepONet vs MLP: {r2_gap:+.2f}% in R²")
    else:
        print("⚠️ Missing baseline results")
    
    print("\n[EXPERIMENT 2] WITH SPATIAL")
    print("-"*90)
    if mlp_spatial and deeponet_spatial:
        print(f"{'Model':<15} {'MAE':<15} {'RMSE':<15} {'R²':<15}")
        print("-"*90)
        print(f"{'MLP':<15} {mlp_spatial['mae']:<15.4f} {mlp_spatial['rmse']:<15.4f} {mlp_spatial['r2']:<15.4f}")
        print(f"{'DeepONet':<15} {deeponet_spatial['mae']:<15.4f} {deeponet_spatial['rmse']:<15.4f} {deeponet_spatial['r2']:<15.4f}")
        
        r2_gap = (deeponet_spatial['r2'] - mlp_spatial['r2']) / mlp_spatial['r2'] * 100
        print(f"\nDeepONet vs MLP: {r2_gap:+.2f}% in R²")
    else:
        print("⚠️ Missing spatial results")
    
    print("\n" + "="*90)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*90)
    
    if mlp_base and deeponet_base and mlp_spatial and deeponet_spatial:
        print(f"{'Experiment':<20} {'MLP R²':<15} {'DeepONet R²':<15} {'DeepONet Gain':<15}")
        print("-"*90)
        
        gain_base = (deeponet_base['r2'] - mlp_base['r2']) / mlp_base['r2'] * 100
        gain_spatial = (deeponet_spatial['r2'] - mlp_spatial['r2']) / mlp_spatial['r2'] * 100
        
        print(f"{'No Spatial':<20} {mlp_base['r2']:<15.4f} {deeponet_base['r2']:<15.4f} {gain_base:>+14.2f}%")
        print(f"{'With Spatial':<20} {mlp_spatial['r2']:<15.4f} {deeponet_spatial['r2']:<15.4f} {gain_spatial:>+14.2f}%")
        
        print("\n" + "="*90)
        print("KEY INSIGHT")
        print("="*90)
        
        if gain_base < 0 and gain_spatial > 0:
            print("✅ PERFECT PATTERN!")
            print(f"   No spatial: MLP wins by {-gain_base:.2f}%")
            print(f"   With spatial: DeepONet wins by {gain_spatial:.2f}%")
            print(f"   → Spatial complexity favors operator learning!")
        elif gain_spatial > gain_base:
            print(f"✅ CLEAR TREND!")
            print(f"   DeepONet advantage increases with complexity:")
            print(f"   No spatial:   {gain_base:+.2f}%")
            print(f"   With spatial: {gain_spatial:+.2f}%")
            print(f"   → Difference: {gain_spatial - gain_base:+.2f} percentage points")
        else:
            print(f"⚠️ Unexpected pattern")
            print(f"   Need more complex scenarios (real data, etc.)")
    
    print("\n" + "="*90)
    print("✓ EXPERIMENTS COMPLETE")
    print("="*90)

if __name__ == '__main__':
    main()
