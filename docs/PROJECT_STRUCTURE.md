# 项目结构说明

## 📁 整理后的项目组织

```
SCM_DeepONet_code/
├── scripts/                              # 所有 Python 脚本存放位置
│   ├── train_mlp_speed.py               # ✅ MLP 训练脚本
│   ├── check_before_training.py         # ✅ 训练前检查脚本
│   ├── run_spatial_comparison.py        # ✅ 数据生成脚本
│   ├── postprocess_with_lags.py         # 基础数据处理
│   ├── postprocess_with_lags_spatial.py # 空间特征数据处理
│   ├── network_spatial_features.py      # 网络拓扑分析
│   └── ...                              # 其他脚本
│
├── data/                                # 数据文件目录
│   ├── dataset_sumo_5km_lag12_no_spatial.npz    # Baseline 数据集
│   ├── dataset_sumo_5km_lag12_with_spatial.npz  # Spatial 数据集
│   └── ...                              # 其他数据文件
│
├── net/                                 # 网络文件
│   └── shanghai_5km.net.xml             # SUMO 网络
│
├── scenarios/                           # SUMO 仿真场景
│   ├── S001/
│   ├── S002/
│   └── ...
│
├── data-3951152/                        # 论文文件
│   └── paper_rev1.tex                   # 论文 LaTeX 文件
│
├── 📚 文档文件 (markdown)
│   ├── QUICK_START.md                   # 快速启动指南
│   ├── TRAINING_GUIDE.md                # 训练脚本使用指南
│   ├── SPATIAL_MODIFICATION_PLAN.md     # 空间特征修改计划
│   ├── ZERO_DATA_FIX.md                 # 零值数据问题说明
│   └── ...
│
└── README.md                            # 项目说明
```

## 🎯 核心执行流程

所有脚本命令都应该从**项目根目录**执行：

```bash
# 1️⃣ 数据生成
python scripts/run_spatial_comparison.py

# 2️⃣ 训练前检查
python scripts/check_before_training.py

# 3️⃣ 训练 Baseline 模型
python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_no_spatial.npz --epochs 100

# 4️⃣ 训练 Spatial 模型
python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_with_spatial.npz --epochs 100
```

## 📝 文件功能说明

### scripts/ 目录

| 文件 | 功能 | 备注 |
|------|------|------|
| `train_mlp_speed.py` | MLP 模型训练 | 输出 MAE/RMSE/R² 指标 + JSON 结果 |
| `check_before_training.py` | 训练前检查 | 验证文件完整性和数据质量 |
| `run_spatial_comparison.py` | 数据生成 | 生成 baseline 和 spatial 数据集 |
| `postprocess_with_lags.py` | 基础数据处理 | 生成 baseline 数据（85维） |
| `postprocess_with_lags_spatial.py` | 空间特征处理 | 生成 spatial 数据（133维） |
| `network_spatial_features.py` | 网络拓扑分析 | 计算上下游邻接特征 |

### 文档说明

| 文件 | 用途 |
|------|------|
| `QUICK_START.md` | 6步完整执行流程 |
| `TRAINING_GUIDE.md` | 训练脚本参数详解 |
| `SPATIAL_MODIFICATION_PLAN.md` | 论文修改模板和位置 |
| `ZERO_DATA_FIX.md` | 零值问题技术说明 |
| `TIME_ANALYSIS.md` | 时间轴分析 |

## ✅ 整理完成检查清单

- [x] `train_mlp_speed.py` 已移到 `scripts/`
- [x] `check_before_training.py` 已移到 `scripts/`
- [x] `run_spatial_comparison.py` 已移到 `scripts/`
- [x] 更新了脚本中的相对路径导入
- [x] 更新了文档中的命令说明
- [x] 删除了根目录下的重复脚本文件

## 🚀 下一步操作

执行以下步骤开始数据生成和模型训练：

```bash
# 从项目根目录执行
cd d:\pro_and_data\SCM_DeepONet_code

# Step 1: 数据生成 (15-25 分钟)
python scripts/run_spatial_comparison.py

# Step 2: 训练前检查 (1 分钟)
python scripts/check_before_training.py

# Step 3: 训练 Baseline (30-80 分钟)
python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_no_spatial.npz --epochs 100

# Step 4: 训练 Spatial (30-80 分钟)
python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_with_spatial.npz --epochs 100
```

详见 `QUICK_START.md`
