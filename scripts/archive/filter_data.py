#!/usr/bin/env python3
# scripts/filter_data.py
# 过滤掉无车流的零值样本

import numpy as np
from pathlib import Path

def filter_data(input_npz, output_npz):
    """保留速度 > 0 的样本"""
    print(f"正在过滤 {input_npz}...")
    
    d = np.load(input_npz, allow_pickle=True)
    X = d["X"]
    Y = d["Y"]
    
    print(f"原始数据: X={X.shape} Y={Y.shape}")
    print(f"原始目标均值: {Y.mean():.6f} km/h")
    
    # 保留速度 > 0 的样本
    mask = Y.reshape(-1) > 0
    X_f = X[mask]
    Y_f = Y[mask]
    
    print(f"过滤后: X={X_f.shape} Y={Y_f.shape}")
    print(f"过滤比例: {len(X_f)/len(X)*100:.1f}%")
    print(f"过滤后目标均值: {Y_f.mean():.6f} km/h")
    
    # 保存
    np.savez_compressed(
        output_npz,
        X=X_f,
        Y=Y_f,
        features=d["features"],
        target=d.get("target"),
        meta=d.get("meta"),
    )
    print(f"✓ 已保存: {output_npz}")
    
    # 验证
    d2 = np.load(output_npz, allow_pickle=True)
    print(f"✓ 验证读取: X={d2['X'].shape} Y={d2['Y'].shape}")
    
    return output_npz

if __name__ == "__main__":
    input_path = "data/dataset_sumo_5km_lag12.npz"
    output_path = "data/dataset_sumo_5km_lag12_filtered.npz"
    
    if not Path(input_path).exists():
        print(f"错误: {input_path} 不存在")
        exit(1)
    
    filter_data(input_path, output_path)
    print("\n过滤完成！现在可以用过滤后的数据训练模型。")
