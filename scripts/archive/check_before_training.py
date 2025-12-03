#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查脚本 - 验证是否可以开始训练
"""

import sys
from pathlib import Path
import json

def check_files():
    """检查必需的文件"""
    print("=" * 70)
    print("📋 检查文件完整性")
    print("=" * 70)
    
    files_to_check = [
        'train_mlp_speed.py',
        'postprocess_with_lags.py',
        'postprocess_with_lags_spatial.py',
        'network_spatial_features.py',
        '../data/dataset_sumo_5km_lag12_no_spatial.npz',
        '../data/dataset_sumo_5km_lag12_with_spatial.npz',
    ]
    
    all_exist = True
    for f in files_to_check:
        p = Path(f)
        exists = p.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {f}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_data_quality():
    """检查数据质量"""
    print("\n" + "=" * 70)
    print("📊 检查数据质量")
    print("=" * 70)
    
    try:
        import numpy as np
        
        for name, file in [
            ('Baseline', '../data/dataset_sumo_5km_lag12_no_spatial.npz'),
            ('Spatial', '../data/dataset_sumo_5km_lag12_with_spatial.npz'),
        ]:
            print(f"\n{name}:")
            
            try:
                data = np.load(file)
                X = data['X']
                Y = data['Y']
                
                print(f"  ✓ 文件可读")
                print(f"    X 形状: {X.shape}")
                print(f"    Y 形状: {Y.shape}")
                
                # 检查 NaN
                has_nan = np.isnan(X).any() or np.isnan(Y).any()
                print(f"    无 NaN: {'✗ 有 NaN' if has_nan else '✓'}")
                
                # 检查全 0 行
                zero_rows = np.sum(np.sum(X, axis=(1, 2)) == 0)
                print(f"    全 0 行数: {zero_rows} {'❌ 警告' if zero_rows > 0 else '✓'}")
                
                # 检查值域
                print(f"    值域: [{X.min():.2f}, {X.max():.2f}]")
                
            except Exception as e:
                print(f"  ✗ 错误: {e}")
    
    except ImportError:
        print("  ✗ numpy 未安装")


def check_dependencies():
    """检查依赖"""
    print("\n" + "=" * 70)
    print("📦 检查依赖")
    print("=" * 70)
    
    dependencies = ['numpy', 'torch']
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} (需要安装: pip install {dep})")


def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                  训练前快速检查                                  ║")
    print("║              Check Before Starting Training                        ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")
    
    files_ok = check_files()
    check_data_quality()
    check_dependencies()
    
    print("\n" + "=" * 70)
    
    if not files_ok:
        print("❌ 某些文件缺失！请先运行：")
        print("   python scripts/run_spatial_comparison.py")
        return False
    
    print("✅ 所有文件就绪！")
    print("\n可以开始训练了！使用以下命令（在根目录执行）：\n")
    
    print("1️⃣  训练 baseline 版本:")
    print("   python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_no_spatial.npz --epochs 100\n")
    
    print("2️⃣  训练 spatial 版本:")
    print("   python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_with_spatial.npz --epochs 100\n")
    
    print("=" * 70)
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
