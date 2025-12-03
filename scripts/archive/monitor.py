#!/usr/bin/env python3
"""
Monitor training progress from log files
"""

import os
import time
import subprocess
from pathlib import Path

def get_latest_epoch(log_file):
    """Extract latest epoch number from log file"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Look for epoch lines (format: Epoch N/100)
        for line in reversed(lines[-100:]):  # Check last 100 lines
            if 'Epoch' in line and '/' in line:
                try:
                    parts = line.split('Epoch')[1].strip().split('/')[0].strip()
                    return int(parts)
                except:
                    pass
        return None
    except:
        return None

def get_final_metrics(log_file):
    """Extract final metrics from completed training"""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Look for final test metrics
        if 'Test MAE' in content:
            lines = content.split('\n')
            mae = rmse = r2 = None
            
            for line in reversed(lines):
                if 'Test MAE' in line and mae is None:
                    try:
                        mae = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'Test RMSE' in line and rmse is None:
                    try:
                        rmse = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'Test R²' in line and r2 is None:
                    try:
                        r2 = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                
                if mae and rmse and r2:
                    break
            
            if mae and rmse and r2:
                return {'mae': mae, 'rmse': rmse, 'r2': r2}
        
        return None
    except:
        return None

def monitor_training():
    """Monitor both trainings"""
    mlp_log = "train_mlp_spatial.log"
    deeponet_log = "train_deeponet_spatial.log"
    
    print("\n" + "="*90)
    print("TRAINING MONITOR")
    print("="*90)
    
    mlp_done = False
    deeponet_done = False
    start_time = time.time()
    
    while True:
        elapsed = int(time.time() - start_time)
        
        mlp_epoch = get_latest_epoch(mlp_log)
        deeponet_epoch = get_latest_epoch(deeponet_log)
        
        mlp_metrics = get_final_metrics(mlp_log)
        deeponet_metrics = get_final_metrics(deeponet_log)
        
        print(f"\n[{elapsed}s elapsed]")
        print("-"*90)
        
        if mlp_epoch:
            print(f"MLP spatial:       Epoch {mlp_epoch}/100", end="")
            if mlp_metrics and not mlp_done:
                print(f" ✓ COMPLETE - R²={mlp_metrics['r2']:.4f}")
                mlp_done = True
            else:
                print(f" ⏳ Training...")
        else:
            print(f"MLP spatial:       Loading data...")
        
        if deeponet_epoch:
            print(f"DeepONet spatial:  Epoch {deeponet_epoch}/100", end="")
            if deeponet_metrics and not deeponet_done:
                print(f" ✓ COMPLETE - R²={deeponet_metrics['r2']:.4f}")
                deeponet_done = True
            else:
                print(f" ⏳ Training...")
        else:
            print(f"DeepONet spatial:  Loading data...")
        
        # Check if both are done
        if mlp_done and deeponet_done:
            print("\n" + "="*90)
            print("✓ BOTH TRAININGS COMPLETE!")
            print("="*90)
            
            print("\nFinal Results:")
            print(f"  MLP spatial:     R²={mlp_metrics['r2']:.4f}, RMSE={mlp_metrics['rmse']:.4f}")
            print(f"  DeepONet spatial: R²={deeponet_metrics['r2']:.4f}, RMSE={deeponet_metrics['rmse']:.4f}")
            
            gain = (deeponet_metrics['r2'] - mlp_metrics['r2']) / mlp_metrics['r2'] * 100
            print(f"\nDeepONet advantage: {gain:+.2f}%")
            
            break
        
        time.sleep(30)  # Check every 30 seconds

if __name__ == '__main__':
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
