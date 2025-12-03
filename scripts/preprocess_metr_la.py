#!/usr/bin/env python3
"""
METR-LA 数据预处理脚本
目的: 从原始METR-LA.h5生成适合MLP和DeepONet训练的数据集
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

def load_metr_la(h5_path):
    """加载METR-LA原始数据"""
    print(f"Loading METR-LA from {h5_path}...")
    
    try:
        df = pd.read_hdf(h5_path)
        speed = df.values
        flow = None # METR-LA usually only has speed in this format
        
        print(f"✓ Speed shape: {speed.shape}")
        return speed, flow
    except Exception as e:
        print(f"Error loading METR-LA: {e}")
        sys.exit(1)

def fill_missing_values(data, method='linear'):
    """填补缺失值"""
    print(f"Filling missing values using {method} method...")
    
    data = data.copy()
    T, N = data.shape
    
    for i in range(N):
        col = data[:, i]
        nan_idx = np.isnan(col)
        
        if method == 'linear':
            # 线性插值
            valid_idx = np.where(~nan_idx)[0]
            if len(valid_idx) < 2:
                # 如果太多缺失，用平均值填充
                col[nan_idx] = np.nanmean(col)
            else:
                col = np.interp(np.arange(T), valid_idx, col[valid_idx])
        elif method == 'forward':
            # 前向填充
            for t in range(T):
                if np.isnan(col[t]) and t > 0:
                    col[t] = col[t-1]
        
        data[:, i] = col
    
    print(f"✓ Missing values filled (max {np.isnan(data).sum()} remaining)")
    return data

# def normalize_data(data, method='z-score'):
#     """标准化数据"""
#     print(f"Normalizing data using {method}...")
#     
#     if method == 'z-score':
#         mean = np.mean(data)
#         std = np.std(data)
#         data = (data - mean) / (std + 1e-8)
#         stats = {'mean': mean, 'std': std}
#     elif method == 'min-max':
#         min_val = np.min(data)
#         max_val = np.max(data)
#         data = (data - min_val) / (max_val - min_val + 1e-8)
#         stats = {'min': min_val, 'max': max_val}
#     
#     print(f"✓ Data normalized: mean={np.mean(data):.4f}, std={np.std(data):.4f}")
#     return data, stats

def create_sequences(data, lag=12, forecast_horizon=1, target_sensor=0):
    """
    创建时间序列数据
    
    Args:
        data: (T, N) 时间步数 × 传感器数
        lag: 历史步数（如12表示过去12个5分钟=1小时）
        forecast_horizon: 预测步数
        target_sensor: 目标预测传感器索引
    
    Returns:
        X: (样本数, 特征数)
        Y: (样本数, 1)
    """
    print(f"\nCreating sequences with lag={lag}, horizon={forecast_horizon}...")
    
    T, N = data.shape
    X_list = []
    Y_list = []
    
    for t in range(lag, T - forecast_horizon):
        # 历史特征: 所有传感器的过去lag步
        x_temporal = data[t-lag:t, :].flatten()  # (lag*N,)
        
        # 目标值: 目标传感器的未来步
        y = data[t + forecast_horizon - 1, target_sensor]
        
        X_list.append(x_temporal)
        Y_list.append(y)
    
    X = np.array(X_list)  # (样本数, lag*N)
    Y = np.array(Y_list)  # (样本数,)
    
    print(f"✓ Created {len(X)} sequences")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  Y statistics: mean={Y.mean():.2f}, std={Y.std():.2f}, min={Y.min():.2f}, max={Y.max():.2f}")
    
    return X, Y

def create_sequences_with_spatial(data, lag=12, forecast_horizon=1, target_sensor=0, n_neighbors=4):
    """
    创建同时包含时间和空间特征的序列
    用于DeepONet: Branch（空间）+ Trunk（时间）
    
    Args:
        data: (T, N) 时间步数 × 传感器数
        lag: 历史步数
        forecast_horizon: 预测步数
        target_sensor: 目标传感器
        n_neighbors: 相邻传感器数
    
    Returns:
        X_temporal: (样本数, lag) - 目标传感器历史
        X_spatial: (样本数, 2*n_neighbors) - 上下游传感器当前状态
        Y: (样本数,)
    """
    print(f"\nCreating sequences WITH spatial features...")
    print(f"  Target sensor: {target_sensor}")
    print(f"  Spatial neighbors: {n_neighbors}")
    
    T, N = data.shape
    X_temporal_list = []
    X_spatial_list = []
    Y_list = []
    
    for t in range(lag, T - forecast_horizon):
        # 时间特征: 目标传感器的历史lag步
        x_temporal = data[t-lag:t, target_sensor]  # (lag,)
        
        # 空间特征: 上游和下游相邻传感器的"过去"状态 (t-1)
        # 注意：不能使用 t 时刻的状态，因为那是我们要预测的时间点 (Leakage Fix)
        upstream_indices = [max(0, target_sensor - i) for i in range(1, n_neighbors + 1)]
        downstream_indices = [min(N-1, target_sensor + i) for i in range(1, n_neighbors + 1)]
        
        x_spatial = np.concatenate([
            data[t-1, upstream_indices],      # 上游传感器速度 (t-1)
            data[t-1, downstream_indices]     # 下游传感器速度 (t-1)
        ])  # (2*n_neighbors,)
        
        # 目标值
        y = data[t + forecast_horizon - 1, target_sensor]
        
        X_temporal_list.append(x_temporal)
        X_spatial_list.append(x_spatial)
        Y_list.append(y)
    
    X_temporal = np.array(X_temporal_list)  # (样本数, lag)
    X_spatial = np.array(X_spatial_list)    # (样本数, 2*n_neighbors)
    Y = np.array(Y_list)                    # (样本数,)
    
    print(f"✓ Created {len(X_temporal)} sequences")
    print(f"  X_temporal (Trunk) shape: {X_temporal.shape}")
    print(f"  X_spatial (Branch) shape: {X_spatial.shape}")
    print(f"  Y shape: {Y.shape}")
    
    return X_temporal, X_spatial, Y

def train_val_test_split(X, Y, X_spatial=None, test_ratio=0.2, val_ratio=0.1):
    """
    时间序列数据划分 (不能随机!)
    先验证，再测试（时间顺序）
    """
    print(f"\nTrain/Val/Test split (time-ordered)...")
    
    N = len(X)
    test_start = int(N * (1 - test_ratio))
    val_start = int(test_start * (1 - val_ratio / (1 - test_ratio)))
    
    X_train = X[:val_start]
    Y_train = Y[:val_start]
    X_val = X[val_start:test_start]
    Y_val = Y[val_start:test_start]
    X_test = X[test_start:]
    Y_test = Y[test_start:]
    
    print(f"Train: {len(X_train)} ({len(X_train)/N*100:.1f}%)")
    print(f"Val:   {len(X_val)} ({len(X_val)/N*100:.1f}%)")
    print(f"Test:  {len(X_test)} ({len(X_test)/N*100:.1f}%)")
    
    if X_spatial is not None:
        X_spatial_train = X_spatial[:val_start]
        X_spatial_val = X_spatial[val_start:test_start]
        X_spatial_test = X_spatial[test_start:]
        
        return (X_train, X_val, X_test, Y_train, Y_val, Y_test,
                X_spatial_train, X_spatial_val, X_spatial_test)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def save_dataset(output_path, X_train, X_val, X_test, Y_train, Y_val, Y_test,
                 X_spatial_train=None, X_spatial_val=None, X_spatial_test=None,
                 feature_names=None, metadata=None):
    """保存为npz格式，适配train_mlp_speed.py的格式"""
    print(f"\nSaving dataset to {output_path}...")
    
    # Concatenate all splits
    X = np.concatenate([X_train, X_val, X_test], axis=0)
    Y = np.concatenate([Y_train, Y_val, Y_test], axis=0)
    
    # Create split indices
    n_train = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)
    
    split = {
        'train': list(range(0, n_train)),
        'val': list(range(n_train, n_train + n_val)),
        'test': list(range(n_train + n_val, n_train + n_val + n_test))
    }
    
    data_dict = {
        'X': X,
        'Y': Y,
        'split': split
    }
    
    if X_spatial_train is not None:
        X_spatial = np.concatenate([X_spatial_train, X_spatial_val, X_spatial_test], axis=0)
        data_dict['X_spatial'] = X_spatial
    
    if feature_names is None:
        # Generate dummy feature names if not provided
        feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    data_dict['features'] = feature_names
    
    if metadata is not None:
        data_dict['meta'] = metadata
    
    np.savez_compressed(output_path, **data_dict)
    print(f"✓ Saved to {output_path}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("="*80)
    print("METR-LA DATA PREPROCESSING")
    print("="*80)
    
    # 路径配置
    input_path = "data/METR-LA.h5"
    output_path_temporal = "data/metr_la_lag12_temporal.npz"
    output_path_spatial = "data/metr_la_lag12_spatial.npz"
    
    # 步骤1: 加载原始数据
    speed, flow = load_metr_la(input_path)
    
    # 步骤2: 填补缺失值
    speed = fill_missing_values(speed, method='linear')
    
    # 步骤3: 标准化 (SKIP - let training script handle it)
    # speed, norm_stats = normalize_data(speed, method='z-score')
    norm_stats = {'mean': 0.0, 'std': 1.0} # Dummy stats
    
    # 步骤4: 创建仅包含时间特征的序列
    print("\n" + "="*80)
    print("Creating TEMPORAL-ONLY sequences")
    print("="*80)
    X_temporal, Y = create_sequences(speed, lag=12, forecast_horizon=1, target_sensor=0)
    
    # 分割数据
    X_train, X_val, X_test, Y_train, Y_val, Y_test = train_val_test_split(
        X_temporal, Y, test_ratio=0.2, val_ratio=0.1
    )
    
    # 保存
    save_dataset(
        output_path_temporal,
        X_train, X_val, X_test, Y_train, Y_val, Y_test,
        metadata={
            'lag': 12,
            'n_sensors': speed.shape[1],
            'target_sensor': 0,
            'normalization': 'z-score',
            'norm_mean': float(norm_stats['mean']),
            'norm_std': float(norm_stats['std']),
        }
    )
    
    # 步骤5: 创建包含空间特征的序列
    print("\n" + "="*80)
    print("Creating TEMPORAL+SPATIAL sequences")
    print("="*80)
    X_temporal_s, X_spatial, Y_s = create_sequences_with_spatial(
        speed, lag=12, forecast_horizon=1, target_sensor=0, n_neighbors=4
    )
    
    # 分割数据
    (X_train_s, X_val_s, X_test_s, Y_train_s, Y_val_s, Y_test_s,
     X_spatial_train, X_spatial_val, X_spatial_test) = train_val_test_split(
        X_temporal_s, Y_s, X_spatial=X_spatial, test_ratio=0.2, val_ratio=0.1
    )
    
    # 保存
    save_dataset(
        output_path_spatial,
        X_train_s, X_val_s, X_test_s, Y_train_s, Y_val_s, Y_test_s,
        X_spatial_train, X_spatial_val, X_spatial_test,
        metadata={
            'lag': 12,
            'n_sensors': speed.shape[1],
            'target_sensor': 0,
            'n_neighbors': 4,
            'normalization': 'z-score',
            'norm_mean': float(norm_stats['mean']),
            'norm_std': float(norm_stats['std']),
        }
    )
    
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\n两个数据集已生成:")
    print(f"  1. {output_path_temporal} - 仅时间特征 (用于MLP, LSTM, TCN)")
    print(f"  2. {output_path_spatial} - 时间+空间特征 (用于GNN, DeepONet)")
    print(f"\n下一步: 训练基线模型")
