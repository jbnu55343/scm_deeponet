#!/usr/bin/env python3
# scripts/diagnose_data.py
# 诊断数据是否与论文一致

import numpy as np
import argparse
from pathlib import Path

def analyze_npz(path):
    """分析 npz 文件的统计特性"""
    print(f"Loading {path}...")
    d = np.load(path, allow_pickle=True)  # 不用 mmap，直接加载
    X = d["X"]
    Y = d["Y"]
    feats = d["features"].tolist()
    print(f"Loaded successfully.")
    
    print(f"\n{'='*80}")
    print(f"NPZ 文件分析: {path}")
    print(f"{'='*80}")
    
    print(f"\n[形状] X={X.shape} Y={Y.shape}")
    print(f"[特征] {feats}")
    print(f"[特征数] {len(feats)}")
    
    # 统计
    print(f"\n{'='*80}")
    print(f"特征统计")
    print(f"{'='*80}")
    
    for i, feat in enumerate(feats):
        x = X[:, i]
        print(f"\n{feat:20s}: "
              f"mean={x.mean():8.4f} std={x.std():8.4f} "
              f"min={x.min():8.4f} max={x.max():8.4f} "
              f"nan={np.isnan(x).sum():6d} inf={np.isinf(x).sum():6d}")
    
    # 目标变量
    y = Y.reshape(-1)
    print(f"\n{'='*80}")
    print(f"目标变量 (speed)")
    print(f"{'='*80}")
    print(f"mean={y.mean():8.4f} std={y.std():8.4f} "
          f"min={y.min():8.4f} max={y.max():8.4f} "
          f"nan={np.isnan(y).sum():6d} inf={np.isinf(y).sum():6d}")
    
    # 过滤分析
    print(f"\n{'='*80}")
    print(f"异常值分析")
    print(f"{'='*80}")
    
    # traveltime 检查
    if "traveltime" in feats:
        idx = feats.index("traveltime")
        tt = X[:, idx]
        valid = (tt > 0) & (tt < 1e5)
        print(f"traveltime > 0 and < 1e5: {valid.sum():,} / {len(tt):,} ({valid.sum()/len(tt)*100:.2f}%)")
    
    # 速度范围
    print(f"speed in reasonable range: {((y >= 0) & (y <= 120)).sum():,} / {len(y):,} ({((y >= 0) & (y <= 120)).sum()/len(y)*100:.2f}%)")
    
    # 论文对标
    print(f"\n{'='*80}")
    print(f"论文对标")
    print(f"{'='*80}")
    print(f"论文数据规模: 23,379,799 (原始) → 1,193,713 (过滤后)")
    print(f"当前数据规模: {X.shape[0]:,} (原始) → ? (过滤后)")
    print(f"论文特征数: 18 input + 1 target = 19 total")
    print(f"当前特征数: {len(feats)}")
    print(f"论文目标: y_mean=8.2547 y_std=10.7359")
    print(f"当前目标: y_mean={y.mean():.4f} y_std={y.std():.4f}")
    
    # 数据质量评分
    print(f"\n{'='*80}")
    print(f"数据质量评分")
    print(f"{'='*80}")
    
    issues = []
    
    # 检查 1: 是否包含 NaN/Inf
    if np.isnan(X).any() or np.isinf(X).any():
        issues.append("❌ X 包含 NaN 或 Inf")
    else:
        print("✅ X 无 NaN/Inf")
    
    if np.isnan(Y).any() or np.isinf(Y).any():
        issues.append("❌ Y 包含 NaN 或 Inf")
    else:
        print("✅ Y 无 NaN/Inf")
    
    # 检查 2: traveltime 过滤
    if "traveltime" in feats:
        idx = feats.index("traveltime")
        tt = X[:, idx]
        valid_ratio = ((tt > 0) & (tt < 1e5)).sum() / len(tt)
        if valid_ratio < 0.04:  # 论文是 5.1%
            issues.append(f"⚠️ traveltime 过滤比例 {valid_ratio*100:.2f}% (论文: 5.1%)")
        else:
            print(f"✅ traveltime 过滤比例 {valid_ratio*100:.2f}% (接近论文 5.1%)")
    
    # 检查 3: 速度范围
    speed_ratio = ((y >= 0) & (y <= 120)).sum() / len(y)
    if speed_ratio < 0.95:
        issues.append(f"⚠️ 速度异常值比例 {(1-speed_ratio)*100:.2f}%")
    else:
        print(f"✅ 速度范围正常 ({speed_ratio*100:.2f}% 在 0-120 km/h)")
    
    # 检查 4: 特征数量
    if len(feats) != 19:
        issues.append(f"⚠️ 特征数 {len(feats)} != 19")
    else:
        print(f"✅ 特征数匹配 (19)")
    
    # 检查 5: 目标变量统计
    if abs(y.mean() - 8.2547) > 1.0:
        issues.append(f"⚠️ y_mean {y.mean():.4f} 与论文 8.2547 差异大")
    else:
        print(f"✅ y_mean 接近论文 ({y.mean():.4f} vs 8.2547)")
    
    if issues:
        print("\n⚠️ 发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 数据质量评分: 优秀 (应该能达到论文性能)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/dataset_sumo_5km_lag12.npz")
    args = ap.parse_args()
    
    analyze_npz(args.npz)
