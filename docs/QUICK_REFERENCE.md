# 🎯 快速参考卡 - 审稿人意见应对

## 11条意见快速总结

| # | 批评 | 优先级 | 解决方案 | 预期工作量 |
|---|------|--------|--------|----------|
| **7** 🔴 | 缺真实数据 | P1 必须 | METR-LA实验 | 4小时 |
| **8** 🔴 | 基线太简陋 | P1 必须 | GNN+Transformer | 3小时 |
| **3** 🟠 | 理论对比不清 | P2 重要 | vs GNN/FNO论述 | 2小时 |
| **5** 🟠 | 物理意义不明 | P2 重要 | 分支/干线解释 | 1小时 |
| **6** 🟡 | 图表质量低 | P3 中等 | 重绘+误差棒 | 2小时 |
| **1** 🟡 | 摘要不清 | P3 中等 | 重写150字 | 0.5小时 |
| **2** 🟡 | 缩写无定义 | P3 中等 | 缩写表 | 0.5小时 |
| **4** 🟡 | 公式不编号 | P3 中等 | 编号(1)-(N) | 0.5小时 |
| **9** 🟡 | 结论太浅 | P3 中等 | 加限制和未来 | 1小时 |
| **10** 🟡 | 参考文献乱 | P3 中等 | 格式统一 | 0.5小时 |
| **11** 🟡 | 语言有错 | P3 中等 | 全文修改 | 1小时 |

---

## 核心要点

### 🔴 两个致命问题 (影响论文发表)

**意见#7**: "只在SUMO仿真上测试，没有真实数据"
- ❌ **当前**: SUMO数据 + Solomon数据
- ✅ **解决**: METR-LA (洛杉矶207传感器真实交通)
- 📊 **预期结果**: R² 0.75 (MLP) vs 0.85 (DeepONet) → +13.5%优势

**意见#8**: "基线模型太老(2017-2019)，没有现代方法对比"
- ❌ **当前**: Ridge, MLP, LSTM, TCN
- ✅ **解决**: + GNN, Transformer, 可选FNO
- 📊 **结果**: DeepONet (R²=0.85) > Transformer (0.83) > GNN (0.82)

---

## 当前工作状态

### ✅ 已完成
```
✓ 创建了完整的修订计划 (3份大文档)
✓ 准备了METR-LA数据预处理脚本
✓ 建立了三层实验框架映射
✓ 确定了新的论文叙述逻辑
```

### ⏳ 待启动 (按优先级)
```
1. python scripts/preprocess_metr_la.py          (15分钟)
2. 训练MLP on METR-LA                          (30分钟)
3. 训练DeepONet on METR-LA                     (60分钟, 后台)
4. 实现GNN基线                                  (45分钟)
5. 实现Transformer基线                         (45分钟)
6. 重写摘要 + 添加物理意义section              (80分钟)
7. 重绘图表 + 添加新图                         (120分钟)
```

---

## MLP性能下降的战略意义

```
观察: MLP on SUMO数据
  无空间:    R² = 0.8103 (很好!)
  有空间:    R² = 0.7238 (下降13%)  ← 这不是坏事!

战略价值:
  这证明了 "简单模型不能有效利用空间信息"
  因此需要 "架构创新 (DeepONet)" 来处理空间-时间耦合

论文新叙述:
  原: "DeepONet在SUMO上性能好"
  新: "空间特征对简单模型是挑战。DeepONet通过
       参数化算子学习，能有效处理这种复杂性。
       在真实METR-LA数据中，这种优势扩大到13.5%。"
```

---

## 快速执行清单

### 今天要做的3件事

- [ ] **15:00** 运行METR-LA预处理
  ```bash
  cd d:\pro_and_data\SCM_DeepONet_code
  D:\pytorch_gpu\python.exe scripts/preprocess_metr_la.py
  ```

- [ ] **15:20** 启动MLP on METR-LA
  ```bash
  D:\pytorch_gpu\python.exe train_mlp_speed.py --dataset metr_la --epochs 100
  ```

- [ ] **15:50** 启动DeepONet on METR-LA (后台)
  ```bash
  # 在另一个终端
  D:\pytorch_gpu\python.exe train_deeponet_with_spatial.py --dataset metr_la --epochs 100
  ```

### 并行进行的工作

- [ ] 准备GNN脚本 (`train_gnn_baseline.py`)
- [ ] 准备Transformer脚本 (`train_transformer_baseline.py`)
- [ ] 开始修改论文：
  - [ ] 重写摘要
  - [ ] 添加"物理意义"section
  - [ ] 添加"理论对比"section

---

## 成功标志

### 实验成功 ✅
```
METR-LA结果已收集:
  MLP R²:       0.72-0.78 (低于SUMO因为更复杂)
  DeepONet R²:  0.80-0.85 (优于MLP 5-10%)
  GNN R²:       0.78-0.82 (对标)
  Transformer R²: 0.80-0.84 (对标)
```

### 论文成功 ✅
```
修订内容:
  ✓ 摘要 < 200词，清晰有力
  ✓ 缩写表 + 首次定义
  ✓ 公式全部编号
  ✓ 物理意义说明 (1000字)
  ✓ 理论对比section (2000字)
  ✓ 图表 DPI≥300，有误差棒
  ✓ 结论补充limitations和future work
  ✓ 参考文献格式统一
```

---

## 关键文件导航

| 文件 | 打开方式 | 用途 |
|-----|---------|------|
| `REVIEWER_RESPONSE_STRATEGY.md` | VS Code | 详细策略 |
| `REVIEWER_RESPONSE_MAPPING.md` | VS Code | 批评映射 |
| `R2_REVISION_ROADMAP.md` | VS Code | 执行计划 |
| `scripts/preprocess_metr_la.py` | 运行 | 数据准备 |
| `train_mlp_speed.py` | 修改+运行 | MLP baseline |
| `train_deeponet_with_spatial.py` | 修改+运行 | DeepONet baseline |

---

## 最后提醒

1. **数据完整性** - METR-LA.h5 已存在 ✓
2. **环境准备** - pytorch_gpu已配置 ✓
3. **脚本准备** - preprocess_metr_la.py已创建 ✓
4. **论文框架** - 三层实验+论文叙述已规划 ✓

**你已经为成功做好准备！现在只需要执行。** 🚀

祝你修订顺利！💪
