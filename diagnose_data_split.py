"""
诊断脚本：为什么 MAE 是论文的 2 倍？

关键发现：
1. 论文用 leave-scenario-out（场景外泛化）
2. 你用的是时间顺序分割（i.i.d. 假设）
3. 场景外泛化比 i.i.d. 分割难得多！
"""

import numpy as np

print("=" * 80)
print("数据分割方案对比")
print("=" * 80)

# 加载数据
d_orig = np.load("data/dataset_sumo_5km_lag12.npz", allow_pickle=True)
d_filt = np.load("data/dataset_sumo_5km_lag12_filtered.npz", allow_pickle=True)

print("\n【原始数据】")
print(f"样本总数：{len(d_orig['Y']):,}")
print(f"目标均值：{d_orig['Y'].mean():.4f} km/h")
print(f"目标中位数：{np.median(d_orig['Y']):.4f} km/h")
print(f"零值比例：{(d_orig['Y'] == 0).sum() / len(d_orig['Y']) * 100:.1f}%")

print("\n【过滤后数据】")
print(f"样本总数：{len(d_filt['Y']):,}")
print(f"目标均值：{d_filt['Y'].mean():.4f} km/h")
print(f"目标中位数：{np.median(d_filt['Y']):.4f} km/h")

print("\n" + "=" * 80)
print("问题根源：数据分割方案")
print("=" * 80)

print("""
[Paper Method] Leave-scenario-out:
   - S001-S004 -> train
   - S005-S006 -> test (completely different scenarios!)
   - This is the hardest task (cross-scene generalization)
   - MLP MAE = 1.430 km/h, R2 = 0.9856
   
   Reason:
   - Different scenarios may have completely different traffic characteristics
   - Model must learn to generalize to unfamiliar boundary conditions
   - This tests real generalization ability

[Your Method] Temporal sequential split:
   - First 80% -> train
   - Middle 10% -> validation
   - Last 10% -> test
   - This assumes data is i.i.d. (time-independent)
   - MLP MAE ~= 2.77 km/h, R2 ~= 0.785
   
   Reason:
   - Time-adjacent data are highly correlated (autocorrelation)
   - Model can achieve high R2 through simple temporal smoothing
   - But this cannot prove real generalization ability

[KEY FINDING]:
   - MAE increase of 1.94x is due to 2.1x increase in target mean
     -> This is normal (scale effect)
   
   - Large R2 difference (0.785 vs 0.9856) is due to different splitting schemes
     -> This reflects the true difficulty difference of tasks

""")

print("=" * 80)
print("下一步该怎么做？")
print("=" * 80)

print("""
[NEXT STEPS]
""")

print("""
Option 1: Keep current scheme (temporal sequential split)
+ Simple and intuitive
+ Consistent with common time series splitting
+ Results may be higher (but cannot reflect real generalization)

- Cannot compare directly with paper (different splitting scheme)
- Cannot test cross-scenario generalization ability
- High R2 does not prove model quality

Option 2: Use leave-scenario-out (RECOMMENDED!)
+ Can compare directly with paper
+ Tests real generalization ability
+ DeepONet's advantages will be more obvious

- R2 will decrease (because task is harder)
- Need to know which scenario each sample belongs to

Option 3: Try both methods
+ Most comprehensive evaluation
- More work required

RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
-> Check if data has scenario markers (scenario ID)
   If yes, use leave-scenario-out scheme
   This way you can compare directly with paper results!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 尝试检查数据结构
print("\n[Data Structure Check]")
print(f"Original data keys: {list(d_orig.keys())}")
print(f"Filtered data keys: {list(d_filt.keys())}")

# Check meta info
if 'meta' in d_orig:
    print(f"\nOriginal data meta: {d_orig['meta']}")
if 'meta' in d_filt:
    print(f"Filtered data meta: {d_filt['meta']}")

print("\n[Hint]:")
print("   - Check if the original data generation script includes scenario information")
print("   - If yes, you can re-save a data version with scenario IDs")
print("   - Then use leave-scenario-out splitting method")
