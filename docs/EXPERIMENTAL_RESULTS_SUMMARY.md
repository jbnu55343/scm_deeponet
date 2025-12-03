# 📊 实验结果汇总表 (Experimental Results Summary)

**最后更新时间**: 2025-12-01 10:30
**状态**: 进行中 (In Progress)

---

## 1. Module 1: SUMO Baseline (No Spatial)
**数据集**: `data/dataset_sumo_5km_lag12_filtered.npz` (19 Features)
**样本量**: ~600k (Train/Val/Test)

| Model | MSE | MAE | RMSE | R2 | Time (s) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Persistence** | 189.0634 | 9.6616 | 13.7500 | -1.0749 | 0.00 | Naive Baseline |
| **Ridge** | 48.9207 | 5.6281 | 6.9943 | 0.4631 | 0.09 | Linear Baseline |
| **MLP** | 18.4547 | 2.8628 | 4.2959 | 0.7975 | 178.42 | Simple NN |
| **TCN** | 19.0868 | 2.9660 | 4.3688 | 0.7905 | 366.18 | Temporal Conv |
| **LSTM** | 16.5115 | 2.5874 | 4.0634 | 0.8188 | 603.20 | RNN Baseline |
| **DeepONet** | 17.1141 | 2.6608 | 4.1369 | 0.8122 | 362.14 | Operator Learning |
| **Transformer** | 16.8369 | 2.6070 | 4.1033 | 0.8152 | 2174.54 | Attention Mechanism |

---

## 2. Module 2: SUMO Spatial (With Spatial)
**数据集**: `data/dataset_sumo_5km_lag12_filtered_with_spatial.npz` (23 Features)
**样本量**: ~600k (Train/Val/Test)
**状态**: 完成 (Completed)

| Model | MSE | MAE | RMSE | R2 | Time (s) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Persistence** | 232.8546 | 11.5196 | 15.2596 | -2.8173 | 0.00 | Naive Baseline |
| **Ridge** | 41.3337 | 5.2291 | 6.4291 | 0.3224 | 0.13 | Linear Baseline |
| **MLP** | 18.1115 | 2.9723 | 4.2558 | 0.7031 | 271.99 | Simple NN |
| **TCN** | 18.4575 | 2.9911 | 4.2962 | 0.6974 | 378.11 | Temporal Conv |
| **GNN (Local)** | 17.2853 | 2.8127 | 4.1576 | 0.7166 | 427.99 | Spatial Baseline |
| **DeepONet** | 15.4123 | 2.6591 | 3.9259 | 0.7473 | 396.57 | Operator Learning |
| **Transformer** | 16.4120 | 2.6997 | 4.0512 | 0.7310 | 2958.43 | Attention Mechanism |
| **LSTM** | 15.3551 | 2.5992 | 3.9186 | 0.7483 | 522.17 | RNN Baseline |

---

## 3. Module 3: METR-LA (Real World)
**数据集**: METR-LA (Graph)
**样本量**: ~34k

| Model | MSE | MAE | RMSE | R2 | Time (s) | Inf Time (s) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Persistence** | 312.55 | 7.61 | 17.68 | 0.3590 | 0.00 | 0.0000 | Naive Baseline |
| **Ridge** | 46.61 | 3.50 | 6.83 | 0.9044 | 2.11 | 0.0285 | Linear Baseline |
| **MLP** | 103.93 | 3.51 | 10.19 | 0.7869 | 5.92 | 0.0840 | Simple NN |
| **TCN** | 51.25 | 4.10 | 7.16 | 0.8949 | 123.74 | 0.3864 | Temporal Conv |
| **LSTM** | 128.01 | 6.47 | 11.31 | 0.7375 | 6.70 | 0.0769 | RNN Baseline |
| **GNN (GCN)** | 51.12 | 4.56 | 7.15 | 0.8952 | 96 | 0.1800 | Graph Baseline |
| **Transformer** | 42.12 | 2.74 | 6.49 | 0.9137 | 73 | 0.3300 | SOTA |
| **DeepONet** | 40.32 | 2.55 | 6.35 | 0.9172 | 92 | 0.0675 | Proposed |

---

## 4. 实验日志 (Experiment Log)

- **2025-12-01 10:00**: 完成 Module 1 (No Spatial) 所有模型训练。LSTM 表现最佳 (R2=0.8188)。
- **2025-12-01 10:30**: 启动 Module 2 (Spatial) 全量模型训练。预计耗时 1-2 小时。
