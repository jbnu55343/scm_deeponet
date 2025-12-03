# 快速参考卡 | Quick Reference Card

## 📋 文件清单 Files Created

```
✅ scripts/network_spatial_features.py
   → 解析 SUMO 网络，构建拓扑

✅ scripts/postprocess_with_lags_spatial.py  
   → 增强数据特征（添加上下游均值）

✅ run_spatial_comparison.py
   → 一键生成对比数据

✅ SPATIAL_MODIFICATION_PLAN.md
   → 详细技术方案（67 行）

✅ PAPER_REVISION_ROADMAP.md
   → 完整实施路线图（本文件）
```

---

## 🚀 快速启动 Quick Start

### 方法 1️⃣：全自动（推荐）
```bash
cd D:\pro_and_data\SCM_DeepONet_code
python run_spatial_comparison.py
```

### 方法 2️⃣：手动两步

**生成数据**：
```bash
# Baseline (无空间特征)
python scripts/postprocess_with_lags.py \
  --scenarios_dir scenarios \
  --out_npz data/dataset_sumo_5km_lag12_no_spatial.npz \
  --features speed entered left density occupancy waitingTime traveltime \
  --lag_features speed \
  --lags 1 2 3 4 5 6 7 8 9 10 11 12 \
  --target speed --horizon 1

# Spatial (有空间特征)
python scripts/postprocess_with_lags_spatial.py \
  --scenarios_dir scenarios \
  --network_file net/shanghai_5km.net.xml \
  --out_npz data/dataset_sumo_5km_lag12_with_spatial.npz \
  --features speed entered left density occupancy waitingTime traveltime \
  --lag_features speed \
  --lags 1 2 3 4 5 6 7 8 9 10 11 12 \
  --target speed --horizon 1 \
  --add_spatial true \
  --spatial_features speed density
```

**训练模型并记录指标**：
```bash
python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_no_spatial.npz
python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_with_spatial.npz
```

---

## 📊 关键指标 Key Metrics to Record

生成的 CSV 或训练输出中，记录这 3 个指标：

| Version | MAE | RMSE | R² |
|---------|-----|------|-----|
| No spatial | ___ | ___ | ___ |
| With spatial | ___ | ___ | ___ |
| Improvement | ___ | ___ | ___ |

---

## 📝 论文修改清单 Checklist for Paper

- [ ] **第 X 段（方法部分）**
  - 添加段落说明 trunk 包含上下游信息
  - 参考文本：见 `SPATIAL_MODIFICATION_PLAN.md`

- [ ] **第 Y 段（实验部分）**
  - 添加 Table X（对比表格）
  - 添加 2-3 句讨论
  - 填入实际指标数据

- [ ] **第 Z 段（局限性部分）**
  - 添加段落承认未用 GNN
  - 说明这是下一步方向
  - 参考文本：见 `SPATIAL_MODIFICATION_PLAN.md`

---

## 🎯 核心回应 Core Arguments to Reviewer

**针对意见 1：真实数据可行性**
```
✅ 已验证 METR-LA 数据集：R² = 0.8333
✅ 这证明 DeepONet 在真实流量数据上有效
✅ 不是 Solomon benchmark 的过度拟合
```

**针对意见 2：空间相关性**
```
✅ 层级 1（数据）：trunk 包含 entered/left/density/occupancy 等聚合量
✅ 层级 2（数据增强）：显式添加上下游邻接特征（speed/density 均值）
✅ 层级 3（论文）：承认 GNN 是下一步，已在 Limitations 中预留
```

---

## 📌 关键文件内容概览

### 1. `network_spatial_features.py` 
**功能**：解析 SUMO .net.xml，构建邻接表
```python
topo = NetworkTopology('net/shanghai_5km.net.xml')
upstream, downstream = topo.get_neighbors('edge_id')
```

### 2. `postprocess_with_lags_spatial.py`
**功能**：读取 edgedata，添加 speed_upstream_mean 等新特征
**参数**：
- `--add_spatial true`
- `--spatial_features speed density`

### 3. 输出文件
```
data/dataset_sumo_5km_lag12_no_spatial.npz      # 基线版本
  → X: (N, 7*12 features)    # 基础 6 维 + 12 步滞后
  → Y: (N, 1)

data/dataset_sumo_5km_lag12_with_spatial.npz    # 增强版本
  → X: (N, 11*12 features)   # 基础 6+4 空间 + 12 步滞后
  → Y: (N, 1)
```

---

## ⏱️ 预计时间 Time Estimate

| 任务 | 时间 |
|------|------|
| 数据生成 (baseline) | 1-2 小时 |
| 数据生成 (spatial) | 1-2 小时 |
| 模型训练 (2 个版本) | 2-4 小时 |
| 论文修改 | 1-2 小时 |
| **总计** | **5-10 小时** |

**加速建议**：
- 可以在数据生成期间阅读文档和准备论文文本
- 使用 GPU 加速训练

---

## 💡 常见陷阱 Common Pitfalls

❌ **不要**：
- 更改网络架构（这是下一步工作）
- 用 GNN（那样就改动太大了）
- 放弃因为性能改进不显著（改进小也能说明问题）

✅ **要**：
- 保持代码简洁（只添加特征，不改模型）
- 记录详细指标（即使差异微小）
- 在论文中坦诚讨论权衡

---

## 🆘 故障排除 Troubleshooting

**问题**：`ModuleNotFoundError: No module named 'network_spatial_features'
```python
# 解决：确保 scripts/ 在 PYTHONPATH
import sys
sys.path.insert(0, '/path/to/scripts')
```

**问题**：XML 解析超慢
```bash
# 原因：file 太大，可以用流式解析
# 已在 network_spatial_features.py 中优化
```

**问题**：特征维度不对
```bash
# 检查：baseline 应该是 6 维 + lag*12
# 检查：spatial 应该是 10 维 + lag*12
```

---

## 📚 进阶参考 Advanced Reference

详细内容参见：
- `SPATIAL_MODIFICATION_PLAN.md` - 技术方案（67 行，包含代码框架）
- `PAPER_REVISION_ROADMAP.md` - 完整路线图（路线图详细）
- 本文档 - 快速参考（5 分钟快速了解）

---

## ✨ 最后提示

这个方案的核心是**有理有据的说法** + **数据支持** + **坦诚的局限性讨论**。

即使审稿人还不满意，你也能自信地说："我们考虑周全，这正是为后续更深入的空间建模研究预留的方向"。

---

**开始吧！祝论文修改顺利！** 🚀📝
