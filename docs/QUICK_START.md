📌 快速参考 - 从数据到论文修改的完整流程
═══════════════════════════════════════════════════════════════════════════════

【当前进度】
───────────────────────────────────────────────────────────────────────────────

✅ Phase 1: 路径问题 - FIXED
✅ Phase 2: 零值数据问题 - FIXED  
✅ Phase 3: 训练脚本 - CREATED
⏳ Phase 4: 数据生成 - PENDING
⏳ Phase 5: 模型训练 - PENDING
⏳ Phase 6: 论文修改 - PENDING


【完整执行流程】(总耗时: 3-4 小时)
───────────────────────────────────────────────────────────────────────────────

Step 1️⃣ : 数据生成 (15-25 分钟)
  命令: python scripts/run_spatial_comparison.py
  输出: 
    • data/dataset_sumo_5km_lag12_no_spatial.npz (baseline)
    • data/dataset_sumo_5km_lag12_with_spatial.npz (spatial)
  验证: 看到日志 "[INFO] S00X: 丢弃前 XXXX 个全 0 时间步"

Step 2️⃣ : 快速检查 (1 分钟)
  命令: python scripts/check_before_training.py
  输出: ✓ 所有文件就绪！或 ✗ 缺少文件

Step 3️⃣ : 训练 Baseline 版本 (30-60 分钟)
  命令: python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_no_spatial.npz --epochs 100
  输出:
    • 训练日志 (Loss 曲线)
    • 测试指标 (MAE, RMSE, R²)
    • JSON 结果文件
  记录: test 行的三个数值 (MAE, RMSE, R²)

Step 4️⃣ : 训练 Spatial 版本 (30-80 分钟)
  命令: python scripts/train_mlp_speed.py --data data/dataset_sumo_5km_lag12_with_spatial.npz --epochs 100
  输出: (同上)
  记录: test 行的三个数值

Step 5️⃣ : 对比性能 (5 分钟)
  查看两个 JSON 文件，比较 test 指标
  计算改进: ΔR² = (spatial_r2 - baseline_r2) × 100

Step 6️⃣ : 修改论文 (1-2 小时)
  编辑: data-3951152/paper_rev1.tex
  位置 1 (方法部分): 说明数据清理和空间特征
  位置 2 (实验部分): 添加 Table X，填入实际数据
  位置 3 (讨论部分): 解释设计选择


【关键脚本速查】
───────────────────────────────────────────────────────────────────────────────

数据生成:          python run_spatial_comparison.py
检查前训练:        python check_before_training.py
训练 baseline:     python train_mlp_speed.py --data data/dataset_sumo_5km_lag12_no_spatial.npz --epochs 100
训练 spatial:      python train_mlp_speed.py --data data/dataset_sumo_5km_lag12_with_spatial.npz --epochs 100
查看结果 (baseline): cat dataset_sumo_5km_lag12_no_spatial_results.json
查看结果 (spatial): cat dataset_sumo_5km_lag12_with_spatial_results.json


【预期结果示例】
───────────────────────────────────────────────────────────────────────────────

Baseline (无空间特征):
  ├─ Train MAE:  3.45 km/h
  ├─ Val MAE:    3.67 km/h
  ├─ Test MAE:   3.78 km/h
  ├─ Test RMSE:  5.63 km/h
  └─ Test R²:    0.8333

Spatial (有空间特征):
  ├─ Train MAE:  3.22 km/h
  ├─ Val MAE:    3.45 km/h
  ├─ Test MAE:   3.58 km/h    (-5.3%)
  ├─ Test RMSE:  5.38 km/h    (-4.4%)
  └─ Test R²:    0.8410       (+0.9%)

对比:
  ├─ 改进: +0.9% R²（说明空间特征有效！）
  ├─ 性能稳定性: spatial 版本略优
  └─ 结论: 空间特征有定量贡献


【关键文档导航】
───────────────────────────────────────────────────────────────────────────────

问题/任务                              参考文档
────────────────────────────────────────────────────────────────────────────
"路径错误"                             README_SOLUTION.md (第 1 层)
"前 0-7.96h 全是 0"                   QUICK_FIX_SUMMARY.md / ZERO_DATA_FIX.md
"怎么训练模型"                         TRAINING_GUIDE.md
"训练脚本用不了"                        TRAINING_GUIDE.md (故障排除)
"模型性能差"                           TRAINING_GUIDE.md (故障排除)
"怎么修改论文"                         SPATIAL_MODIFICATION_PLAN.md
"3 层防御战略是什么"                   PAPER_REVISION_ROADMAP.md / FINAL_SUMMARY.md
"我什么时候开始"                       START_HERE.txt / QUICK_REFERENCE.md


【论文修改模板】(已准备)
───────────────────────────────────────────────────────────────────────────────

方法部分 (120 字):
  在 Methods 中添加段落说明空间特征设计和数据清理
  模板: SPATIAL_MODIFICATION_PLAN.md 第 3 部分

实验部分 (Table + 3 句讨论):
  在 Experiments/Results 中添加对比表格
  表格模板: SPATIAL_MODIFICATION_PLAN.md 第 3 部分
  讨论模板: SPATIAL_MODIFICATION_PLAN.md 第 3 部分

局限性部分 (100 字):
  在 Conclusions 中添加段落说明设计选择
  模板: SPATIAL_MODIFICATION_PLAN.md 第 3 部分


【常见问题速解】
───────────────────────────────────────────────────────────────────────────────

Q: 我应该先做什么？
A: python run_spatial_comparison.py (生成数据)

Q: 如何检查数据已生成？
A: ls data/dataset_sumo_5km_lag12_*.npz

Q: PyTorch 怎么装？
A: pip install torch 或参考 TRAINING_GUIDE.md

Q: 训练需要多久？
A: GPU 5-12 分钟/个模型；CPU 30-80 分钟/个模型

Q: 结果保存在哪？
A: 每次训练生成 {data_file_stem}_results.json

Q: 如何提取 test 指标？
A: 打开 JSON 文件，找 "test" 字段下的 mae/rmse/r2

Q: 空间特征的改进不明显怎么办？
A: 即使只有 0.5-1% 的改进也足以论证有效性！

Q: 如何修改论文？
A: 参考 SPATIAL_MODIFICATION_PLAN.md 第 3-4 部分


【重要提醒】
───────────────────────────────────────────────────────────────────────────────

✓ 数据已自动过滤全 0 时间步 (0-8h)
✓ 训练脚本自动处理 NaN 和标准化
✓ 结果自动保存为 JSON （用于论文）
✓ 最佳模型自动选择 (基于验证集)

❌ 不要修改脚本参数（除非你知道后果）
❌ 不要中断训练（会丢失结果）
❌ 不要删除生成的 .npz 文件


【最终检查清单】
───────────────────────────────────────────────────────────────────────────────

□ 数据生成完毕 (两个 .npz 文件存在)
□ 日志显示丢弃了全 0 时间步
□ Baseline 训练完毕 (test 指标已记录)
□ Spatial 训练完毕 (test 指标已记录)
□ 两个 JSON 结果文件存在
□ 计算了性能改进
□ 论文 Methods 已修改
□ 论文 Experiments 已修改 (添加 Table)
□ 论文 Conclusions 已修改
□ LaTeX 编译无错误


═══════════════════════════════════════════════════════════════════════════════

立即开始：

  python run_spatial_comparison.py

然后按照上面的 6 个步骤依次执行！

═══════════════════════════════════════════════════════════════════════════════
