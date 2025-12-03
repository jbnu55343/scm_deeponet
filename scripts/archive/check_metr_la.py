#!/usr/bin/env python3
"""
METR-LA 数据集加载和快速诊断
目的: 验证METR-LA数据是否可用，准备用于模型训练
"""

import os
import numpy as np
import h5py
from pathlib import Path

print("="*80)
print("METR-LA 数据集诊断")
print("="*80)

# ============================================================================
# 检查数据文件是否存在
# ============================================================================
print("\n[1] 检查数据文件...")

possible_paths = [
    "data/METR-LA.h5",
    "data/metr_la/METR-LA.h5",
    "data/METR-LA/METR-LA.h5",
    "D:/data/METR-LA.h5",
    "../data/METR-LA.h5",
]

data_path = None
for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        print(f"✓ 找到数据文件: {path}")
        break

if data_path is None:
    print("⚠ 未找到METR-LA.h5文件")
    print("\n可能的位置:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\n请下载METR-LA数据集:")
    print("  来源: http://pems.dot.ca.gov/")
    print("  或: GitHub METR-LA benchmark")
    print("\n如果已下载，请手动指定路径...")
    data_path = None
else:
    # ============================================================================
    # 加载并检查数据
    # ============================================================================
    print("\n[2] 加载METR-LA数据...")
    
    try:
        with h5py.File(data_path, 'r') as f:
            # 检查数据集结构
            print(f"\n数据集keys: {list(f.keys())}")
            
            # 通常METR-LA有'speed'、'flow'等字段
            if 'speed' in f:
                data = f['speed'][:]
                print(f"\n✓ 找到'speed'数据")
                print(f"  形状: {data.shape}")
                print(f"  数据类型: {data.dtype}")
                print(f"  统计: mean={np.nanmean(data):.2f}, std={np.nanstd(data):.2f}, min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}")
                
                # 检查缺失值
                missing_ratio = np.sum(np.isnan(data)) / data.size
                print(f"  缺失值比例: {missing_ratio*100:.2f}%")
                
            else:
                print("\n数据字段可用:")
                for key in f.keys():
                    print(f"  - {key}: {f[key].shape}")
                    
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        data_path = None

# ============================================================================
# 如果成功加载，生成数据统计
# ============================================================================
if data_path:
    print("\n[3] 数据准备建议...")
    print("""
    对于METR-LA数据的处理流程：
    
    1. 数据预处理:
       - 填补缺失值 (线性插值或前值填充)
       - 标准化 (Z-score: (x-mean)/std)
       - 序列化 (时间窗口: 12步历史 → 1步预测)
    
    2. 训练-验证-测试划分:
       - 时间序列不能随机划分!
       - 建议: 前60% train, 20% val, 20% test
       - 或: 按时间分割(如按周划分)
    
    3. 特征工程:
       - 使用历史速度: y_{t-1}, y_{t-2}, ..., y_{t-12}
       - 如果有相邻传感器信息: 相邻节点的速度/流量
       - 可选: 时间特征(小时、周几、是否周末)
    
    4. 模型输入:
       - MLP: 所有特征连接 (12维时间 + N维空间 = X)
       - DeepONet:
         - Branch: 相邻传感器特征 (4维空间特征)
         - Trunk: 历史速度 (12维时间特征)
       - Transformer: 所有特征 (结合注意力机制)
       - GNN: 图结构 + 节点特征
    """)

print("\n" + "="*80)
print("下一步行动:")
print("="*80)
print("""
1. 确保METR-LA.h5文件在正确位置
2. 如果有数据，运行以下脚本进行预处理:
   python scripts/preprocess_metr_la.py
3. 然后训练基线模型:
   python scripts/train_mlp_speed.py --dataset metr_la --epochs 100
4. 再训练DeepONet:
   python train_deeponet_with_spatial.py --dataset metr_la --epochs 100
""")
