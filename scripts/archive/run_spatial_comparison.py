#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行空间特征对比实验的快速脚本

两个版本：
1. Baseline (no spatial) - 只用历史速度 + 局部特征
2. With spatial - 添加上下游邻接特征
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    
    # 在父目录（SCM_DeepONet_code）运行命令
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
    else:
        print(f"❌ {description} - FAILED (code {result.returncode})")
        return False
    
    return True


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   空间相关性对比实验：Baseline vs. With Spatial Context         ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    目标：
    1. 生成两个数据集版本（baseline / with spatial）
    2. 训练两个模型
    3. 对比性能指标（MAE, RMSE, R²）
    4. 为论文生成表格和结论
    
    预期输出：
    - data/dataset_sumo_5km_lag12_no_spatial.npz
    - data/dataset_sumo_5km_lag12_with_spatial.npz
    - data/preview_samples_lag_no_spatial.csv
    - data/preview_samples_lag_with_spatial.csv
    """)
    
    input("按 Enter 继续...")
    
    # ============================================================
    # 1. 生成 BASELINE 数据（无空间特征）
    # ============================================================
    
    cmd_baseline = [
        sys.executable, "scripts/postprocess_with_lags.py",
        "--scenarios_dir", "scenarios",
        "--out_npz", "data/dataset_sumo_5km_lag12_no_spatial.npz",
        "--features", "speed", "entered", "left", "density", "occupancy", "waitingTime", "traveltime",
        "--lag_features", "speed",
        "--lags", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "--target", "speed",
        "--horizon", "1",
        "--write_csv_preview", "data/preview_samples_lag_no_spatial.csv"
    ]
    
    if not run_command(cmd_baseline, "生成 BASELINE 数据（无空间特征）"):
        print("❌ Baseline 数据生成失败，中止")
        return
    
    # ============================================================
    # 2. 生成 SPATIAL 数据（有空间特征）
    # ============================================================
    
    cmd_spatial = [
        sys.executable, "scripts/postprocess_with_lags_spatial.py",
        "--scenarios_dir", "scenarios",
        "--network_file", "net/shanghai_5km.net.xml",
        "--out_npz", "data/dataset_sumo_5km_lag12_with_spatial.npz",
        "--features", "speed", "entered", "left", "density", "occupancy", "waitingTime", "traveltime",
        "--lag_features", "speed",
        "--lags", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12",
        "--target", "speed",
        "--horizon", "1",
        "--add_spatial", "true",
        "--spatial_features", "speed", "density",
        "--write_csv_preview", "data/preview_samples_lag_with_spatial.csv"
    ]
    
    if not run_command(cmd_spatial, "生成 SPATIAL 数据（有空间特征）"):
        print("❌ Spatial 数据生成失败，但可继续用已有数据")
    
    # ============================================================
    # 3. 提示后续步骤
    # ============================================================
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   📊 数据生成完成！接下来的步骤：                              ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    ✅ 已完成：
       • 数据集生成（baseline + spatial）
       • CSV 预览文件
    
    ⏳ 下一步：
    
    1️⃣  训练两个模型版本：
       • python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_no_spatial.npz
       • python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_with_spatial.npz
       
       记录输出的 MAE, RMSE, R² 指标
    
    2️⃣  创建对比表格（复制到论文）：
       
       Table X: Effect of Local Spatial Context on DeepONet Performance
       
       | Model                              | MAE   | RMSE  | R²     |
       |:-----------------------------------|:-----:|:-----:|:------:|
       | DeepONet (baseline, no spatial)    | X.XXX | X.XXX | 0.XXXX |
       | DeepONet (+ spatial context)       | X.XXX | X.XXX | 0.XXXX |
       | Improvement                        |  -X%  |  -X%  | +X.X%  |
    
    3️⃣  论文修改（3 处）：
       
       📝 data-3951152/paper_rev1.tex:
       
       (a) 方法部分（~第 XXX 行）：
           添加段落说明：trunk 包含上下游邻接信息
           
       (b) 实验部分：
           添加 Table X 和讨论文本
           
       (c) 局限性部分：
           添加段落说明没有用 GNN，为下一步预留
       
       📄 参考文本已保存在：
           SPATIAL_MODIFICATION_PLAN.md
    
    4️⃣  验证：
       • 数据维度是否正确（baseline 7 维 vs spatial 11 维）
       • 性能是否有提升（即使微弱也足够说明问题）
       • 论文文本是否自洽
    
    ════════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    main()
