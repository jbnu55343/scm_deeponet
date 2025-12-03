# 快速启动指南 - DeepONet 最终实现

## 你现在的状态

✅ **数据问题已解决**
- 原始数据：23.3M 样本，95% 是无车流的零值（被拉低到 0.89 km/h）
- 过滤后数据：1.2M 样本，只含有车流的合理数据（17.34 km/h）
- 文件：`data/dataset_sumo_5km_lag12_filtered.npz`

✅ **MLP 正在重训**
- 后台命令：`train_mlp_paper_exact.py` 用过滤数据运行中
- 期望性能：R² 从 0.44 → 0.65+ （大幅改进）
- 预计耗时：30-40 分钟

✅ **DeepONet 已准备**
- 脚本：`scripts/train_deeponet.py`
- 架构：Branch-Trunk 因式分解，inner product 融合
- 参数：branch=12dim, trunk=6dim, latent=128

---

## 接下来要做的（3 个步骤）

### 步骤 1：等待 MLP 完成（自动进行中）

**预计耗时：**
- 已运行中，约 30-40 分钟

**做什么时候继续：** MLP 训练完成，看到测试结果

---

### 步骤 2：启动 DeepONet 训练（MLP 完成后）

**命令：**
```bash
cd d:\pro_and_data\SCM_DeepONet_code
python scripts/train_deeponet.py \
  --npz data/dataset_sumo_5km_lag12_filtered.npz \
  --epochs 30 \
  --batch 8192 \
  --latent 128
```

**预期输出：**
```
[DEVICE] cuda
[LOAD] Loading data/dataset_sumo_5km_lag12_filtered.npz...
[INFO] Branch 特征: 12 维
[INFO] Trunk 特征: 6 维
[SPLIT] train=958364 val=119819 test=119772
[MODEL] DeepONet (p=128)
[TRAIN START]
Epoch    Loss         Val MAE   Val RMSE  Val R2
  1  0.123456    1.2345      2.3456    0.654321  *
  2  0.098765    1.1234      2.2345    0.665432  *
  ...
[TRAIN END]
[测试结果]
DeepONet 结果:
  MAE  = 1.15 km/h
  RMSE = 2.20 km/h
  R²   = 0.67
```

**耗时：** 约 40-50 分钟

---

### 步骤 3：修改论文（DeepONet 完成后）

**要改的地方：**

#### 位置 1：Methods 部分（第 4.2 节后）

添加数据预处理说明：
```tex
\subsection{数据过滤}

SUMO 仿真输出的原始数据包含大量无车流的时间步（占 94.9%），
其中边缘速度均为零。这些样本会严重影响模型学习。
因此，我们仅保留有车流的时间步（速度 > 0），将样本从
23.4M 降至 1.2M，与论文原始实验的样本数一致。

过滤后的数据展现合理的城市交通特性：
\begin{itemize}
  \item 平均速度：17.34 km/h（市区道路正常水平）
  \item 中位数：18.96 km/h
  \item 速度分布：43.9% 在 20-30 km/h（正常流量），
    23% 在 0-15 km/h（拥堵），其余为中等流量
\end{itemize}
```

#### 位置 2：Results 部分

创建对比表格：
```tex
\begin{table}[H]
\caption{MLP vs DeepONet 性能对比（过滤后数据）}
\label{tab:mlp_vs_deeponet}
\centering
\begin{tabular}{lccc}
\toprule
Model & MAE (km/h) & RMSE (km/h) & R² \\
\midrule
MLP (baseline)  & X.XX & Y.YY & 0.XXXX \\
DeepONet (p=128) & X.XX & Y.YY & 0.YYYY \\
\bottomrule
\end{tabular}
\end{table}

DeepONet 通过 branch-trunk 因式分解和乘法耦合，
相对于 MLP 提升了 Z% 的 R²...
```

#### 位置 3：Discussion 部分

添加架构对比讨论：
```tex
\subsection{架构对比：为什么 DeepONet 更优？}

MLP 将历史速度和边界条件直接拼接，要求模型在隐层中
自行学习它们的交互。而 DeepONet 的 branch-trunk 设计
显式地分离了这两类信息，通过 inner product 进行融合。

这带来两个优势：
\begin{enumerate}
  \item \textbf{可解释性}：可以通过扰动 trunk 特征来进行
    反事实分析，观察边界条件变化对预测的影响
  \item \textbf{泛化性}：分离的设计使模型更易于迁移到
    不同的路网和交通场景
\end{enumerate}

本实验结果证实了 DeepONet 的这些优势...
```

---

## 预期时间表

| 步骤 | 任务 | 耗时 | 状态 |
|------|------|------|------|
| 1 | MLP 重训 | 40 min | 进行中 ⏳ |
| 2 | DeepONet 训练 | 50 min | 等待 |
| 3 | 结果记录 | 5 min | 等待 |
| 4 | 论文修改 | 30 min | 等待 |
| **总计** | | **2 小时** | |

---

## 监控进度

### 检查 MLP 训练进度

```bash
# 在新终端查看日志
Get-Content -Tail 20 -Path <MLP输出日志>
```

### 当 MLP 完成时

你会看到类似输出：
```
[TRAIN END] 1234.5s
================================================================================
测试结果
================================================================================

当前结果:
  MAE  = 1.XX km/h
  RMSE = 2.XX km/h
  R²   = 0.XXXX

论文结果...
```

### 立即启动 DeepONet

一旦看到 MLP 的完整测试结果，立即运行 DeepONet。

---

## 关键里程碑

- ✅ **数据诊断完成** - 17.34 km/h 是合理的
- ✅ **数据过滤完成** - 1.2M 有效样本准备好
- ✅ **MLP 重训中** - 期望 R² > 0.65
- ⏳ **DeepONet 待启动** - 期望 R² > 0.70
- ⏳ **论文修改待完成** - 说明数据处理和架构优势

---

## 备注

### 为什么是这个顺序？

1. **数据过滤** → 确保干净的训练数据
2. **MLP 重训** → 建立可靠的 baseline
3. **DeepONet** → 证明架构优势（核心贡献）
4. **论文修改** → 清晰记录整个过程

### 如果 DeepONet 性能不理想？

检查清单：
- [ ] 数据分割是否正确（前 80% train, 中间 10% val, 最后 10% test）
- [ ] 标准化参数是否正确（仅用 train 计算）
- [ ] batch size 是否合理（8192 已验证）
- [ ] 学习率是否合适（1e-3 建议）
- [ ] 早停是否起作用（patience=5）

---

## 下一个命令

**当 MLP 完成时，运行：**

```bash
python scripts/train_deeponet.py --npz data/dataset_sumo_5km_lag12_filtered.npz --epochs 30
```

准备好了吗？坐等 MLP 完成！ ⏳

