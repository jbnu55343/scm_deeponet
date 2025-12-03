# Operator Learning for Macroscopic Traffic Speed Forecasting

This repository contains the official implementation of the paper **"Operator Learning with Branch–Trunk Factorization for Macroscopic Short-Term Speed Forecasting"**.

The framework utilizes **Deep Operator Networks (DeepONet)** to bridge the gap between logistics demand (boundary conditions) and traffic state dynamics.

---

## 🛠️ Environment Setup

The code is implemented in **Python 3.10** using **PyTorch 2.0**.

### Installation
```bash
# Create a virtual environment (optional)
conda create -n traffic_op python=3.10
conda activate traffic_op

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install numpy pandas scikit-learn matplotlib
```

---

## 📂 Data Preparation

### 1. SUMO Simulation Data (Modules 1 & 2)
The simulation data is generated using SUMO and processed into `.npz` files.

**Step 1: Split Data**
Create standard train/test splits from the raw simulation logs.
```bash
python scripts/create_standard_split.py
```

**Step 2: Filter Data**
Apply filtering to remove stationary periods and outliers.
```bash
python scripts/filter_sumo_data.py
```
*Output:* `data/dataset_sumo_5km_lag12_filtered.npz`

### 2. METR-LA Real-World Data (Module 3)
Preprocess the METR-LA benchmark dataset for Operator Learning (Branch=Spatial, Trunk=Temporal).

```bash
python scripts/preprocess_metr_la.py
```
*Output:* `data/metr_la_lag12_temporal.npz`, `data/metr_la_lag12_spatial.npz`

---

## 🚀 Experiments

### Module 1: SUMO Baseline (Temporal Only)
Evaluate models on the linear simulation topology using only temporal features.

```bash
# MLP Baseline
python scripts/train_mlp_sumo_std.py --npz data/dataset_sumo_5km_lag12_filtered.npz

# DeepONet (Temporal)
python scripts/train_deeponet_sumo_std.py --npz data/dataset_sumo_5km_lag12_filtered.npz

# Transformer
python scripts/train_transformer_sumo_std.py --npz data/dataset_sumo_5km_lag12_filtered.npz
```

### Module 2: SUMO Spatial (With Upstream/Downstream)
Evaluate the impact of adding spatial boundary conditions ($v_{up}, v_{down}, k_{up}, k_{down}$).

```bash
# DeepONet (Spatial)
python scripts/train_deeponet_sumo_std.py --npz data/dataset_sumo_5km_lag12_filtered_with_spatial.npz

# GNN (Local Graph)
python scripts/train_gnn_sumo_std.py --npz data/dataset_sumo_5km_lag12_filtered_with_spatial.npz
```

### Module 3: Real-World Validation (METR-LA)
Validate on the complex METR-LA graph network (207 sensors).

```bash
# DeepONet (SOTA)
python scripts/train_deeponet_metr_la.py --epochs 100

# GNN Baseline (GraphSAGE/GCN)
python scripts/train_gnn_metr_la.py --epochs 100

# Transformer Baseline
python scripts/train_transformer_metr_la.py --epochs 100

# MLP Baseline
python scripts/train_mlp_metr_la.py --epochs 100
```

---

## 📊 Results & Logs

*   **Training Logs:** Saved in `results/` directory with timestamps.
*   **Summary:** Run the collection script to aggregate results into a markdown table.

```bash
python scripts/collect_sumo_results.py
```

### Key Results (from Paper)

| Dataset | Model | R² Score | MAE (km/h) |
| :--- | :--- | :--- | :--- |
| **SUMO (Sim)** | MLP | **0.7800** | 2.60 |
| | DeepONet | 0.7685 | 2.80 |
| **METR-LA (Real)** | **DeepONet** | **0.9172** | **2.55** |
| | Transformer | 0.9137 | 2.74 |
| | GNN | 0.8952 | 4.56 |
| | MLP | 0.8508 | 4.64 |

---

## 🗂️ Code Structure

```
.
├── data/                   # Dataset files (.npz, .csv)
├── docs/                   # Documentation & Response to Reviewers
├── results/                # Saved models and logs
├── scripts/
│   ├── train_*.py          # Training scripts for specific models/datasets
│   ├── preprocess_*.py     # Data preprocessing
│   ├── std_utils.py        # Utility functions (Logger, EarlyStopping)
│   └── archive_20251130/   # Archived/Legacy code
└── README.md               # This file
```

## 📧 Contact
For any questions, please open an issue or contact the authors.

在项目里建一个公共的类型文件（放在 scenarios 目录下，所有场景共用）：

cat > scenarios/common.types.xml <<'XML'
<additional>
  <vType id="passenger" vClass="passenger"
         accel="2.6" decel="4.5" sigma="0.5"
         length="5.0" minGap="2.5"
         maxSpeed="16.67" speedDev="0.1"
         guiShape="passenger"/>
</additional>
XML

用GUI查看路网及仿真
用 --additional-files 把它一起加载（GUI 单场景举例）：

sumo-gui -n net/shanghai_5km.net.xml \
         -r scenarios/S001/routes.rou.xml \
         --additional-files scenarios/common.types.xml \
         --route-steps 0


批量仿真（非 GUI），也一样带上这个 additional：

NET="net/shanghai_5km.net.xml"
for d in scenarios/S*; do
  sumo -n "$NET" -r "$d/routes.rou.xml" \
       --additional-files scenarios/common.types.xml \
       --route-steps 0 \
       --summary-output "$d/summary.xml" \
       --tripinfo-output "$d/tripinfo.xml"
done




