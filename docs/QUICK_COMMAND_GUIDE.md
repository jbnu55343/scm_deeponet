# 快速命令指南

## 环境设置问题已解决 ✅

**问题**：VS Code 终端中无法激活 PyTorch 虚拟环境
**解决方案**：直接使用 PyTorch 环境的 Python 可执行文件路径

---

## 快速执行指令

在 VS Code 的 PowerShell 终端中运行以下命令：

### 1. 数据生成（必须第一步）
```powershell
D:\DL\envs\pytorch_gpu\python.exe scripts/run_spatial_comparison.py
```
**耗时**：15-25 分钟  
**预期输出**：`[INFO] S00X: 丢弃前XXXX个全0时间步` 日志  
**生成文件**：
- `data/dataset_sumo_5km_lag12_no_spatial.npz`（基础版本）
- `data/dataset_sumo_5km_lag12_with_spatial.npz`（空间增强版本）

---

### 2. 基础模型训练（无空间特征）
```powershell
D:\DL\envs\pytorch_gpu\python.exe scripts/train_mlp_speed.py --npz data/dataset_sumo_5km_lag12_no_spatial.npz --epochs 100
```
**耗时**：30-80 分钟  
**记录指标**：最终输出的 MAE, RMSE, R² 值（用于对比）  
**保存模型**：`models/mlp_speed.pt`

---

### 3. 空间模型训练（含空间特征）
```powershell
D:\DL\envs\pytorch_gpu\python.exe scripts/train_mlp_speed.py --npz data/dataset_sumo_5km_lag12_with_spatial.npz --epochs 100
```
**耗时**：30-80 分钟  
**预期改进**：R² 应该比基础模型高 0.5-1%  
**记录指标**：最终输出的 MAE, RMSE, R² 值

---

## 脚本参数说明

### `train_mlp_speed.py` 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--npz` | `data/dataset_sumo_5km.npz` | **必需**：输入数据文件路径（注意参数名是 `--npz` 不是 `--data`） |
| `--epochs` | `10` | 训练轮数 |
| `--batch` | `8192` | 批次大小 |
| `--lr` | `0.001` | 学习率 |
| `--hidden` | `256,256` | 隐藏层维度（逗号分隔） |
| `--dropout` | `0.1` | Dropout 比例 |
| `--save` | `models/mlp_speed.pt` | 模型保存路径 |

### 完整示例
```powershell
D:\DL\envs\pytorch_gpu\python.exe scripts/train_mlp_speed.py `
  --npz data/dataset_sumo_5km_lag12_no_spatial.npz `
  --epochs 100 `
  --batch 8192 `
  --lr 0.001 `
  --hidden 256,256 `
  --dropout 0.1
```

---

## 常见错误与解决

### 错误 1: `ModuleNotFoundError: No module named 'torch'`
**原因**：没有使用 PyTorch 环境的 Python  
**解决**：使用完整路径 `D:\DL\envs\pytorch_gpu\python.exe`

### 错误 2: `unrecognized arguments: --data`
**原因**：参数名错误（使用了 `--data` 而不是 `--npz`）  
**解决**：改用 `--npz` 参数

### 错误 3: `FileNotFoundError: data/dataset_sumo_5km_lag12_no_spatial.npz`
**原因**：数据文件不存在  
**解决**：先运行数据生成脚本（步骤1）

---

## 执行流程

```
1. 数据生成 (15-25 min)
   ↓
2. 基础模型训练 (30-80 min) → 记录指标
   ↓
3. 空间模型训练 (30-80 min) → 记录指标
   ↓
4. 对比与论文修改 (1-2 hours)
   - 编辑 data-3951152/paper_rev1.tex
   - 3处修改位置（模板见 SPATIAL_MODIFICATION_PLAN.md）
```

---

## 文件位置查询

| 项目 | 路径 |
|------|------|
| 训练脚本 | `scripts/train_mlp_speed.py` |
| 数据生成脚本 | `scripts/run_spatial_comparison.py` |
| 生成的数据 | `data/dataset_sumo_5km_lag12_*.npz` |
| 论文文件 | `data-3951152/paper_rev1.tex` |
| 论文修改模板 | `SPATIAL_MODIFICATION_PLAN.md` |

---

## 环境验证

如果需要验证 PyTorch 环境是否正常：
```powershell
D:\DL\envs\pytorch_gpu\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

预期输出：
```
PyTorch: 2.9.0.dev20250712+cu129
CUDA: True
```

---

最后更新：2025-11-29
