
---

## EXPERIMENT_LOG.md 模板（每次一节）
```markdown
## 2025-08-13 Exp-01
- 中心/半径：人民广场，5km
- OSM：overpass bbox=...
- netconvert：--proj utm,51N ... 生成 net/shanghai_5km.net.xml
- 吸附：fit_margin=150, max_radius=2000, grow=1.6 → snapped=101/101
- trips：S001..S006 base=5000 multipliers=[...] 窗口=[...]
- duarouter：routes 车辆数 S001=2243, S002=...
- 仿真：route-steps=0, edgeData freq=60，输出到 scenarios/*/outputs
- 数据集：U=flow, Y=speed，S×E×T=...
- 结果：DeepONet test RMSE=..., 基线(MLP/TCN)=...
- 发现/修复：未排序忽略→route-steps=0；vType=passenger→vtypes.add.xml
- 产物：data/dataset_traffic_5km.npz, models/exp01.ckpt, figs/...




20250820    已能够在GUI中将数据跑起来，下一步是用仿真生成数据以及构建数据集