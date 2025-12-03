# -*- coding: utf-8 -*-
"""
postprocess_with_lags_spatial.py

改进版本：支持添加上下游空间特征（速度、密度等均值）

用法：
python scripts/postprocess_with_lags_spatial.py \
  --scenarios_dir scenarios \
  --network_file net/shanghai_5km.net.xml \
  --out_npz data/dataset_sumo_5km_lag12_with_spatial.npz \
  --features speed entered left density occupancy waitingTime traveltime \
  --lag_features speed \
  --lags 1 2 3 4 5 6 7 8 9 10 11 12 \
  --target speed \
  --horizon 1 \
  --add_spatial true \
  --spatial_features speed density \
  --write_csv_preview data/preview_samples_lag_spatial.csv
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# 导入网络拓扑模块
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from network_spatial_features import NetworkTopology
except ImportError:
    print("[WARN] Could not import NetworkTopology, spatial features will be disabled")
    NetworkTopology = None


def parse_args():
    p = argparse.ArgumentParser("Build dataset with spatial-aware lags from edgedata.xml")
    p.add_argument("--scenarios_dir", type=str, default="scenarios")
    p.add_argument("--network_file", type=str, default="net/shanghai_5km.net.xml")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="基础特征名（来自 edgedata.xml）",
    )
    p.add_argument(
        "--lag_features",
        nargs="+",
        default=[],
        help="需要做滞后的特征子集",
    )
    p.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 2],
        help="时间滞后阶数",
    )
    p.add_argument("--target", type=str, default="speed")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--write_csv_preview", type=str, default="")
    
    # 新增：空间特征选项
    p.add_argument(
        "--add_spatial",
        type=str,
        default="false",
        choices=["true", "false"],
        help="是否添加上下游空间特征",
    )
    p.add_argument(
        "--spatial_features",
        nargs="+",
        default=["speed", "density"],
        help="要聚合的空间特征名",
    )
    
    return p.parse_args()


def read_edgedata_xml(path_xml: str, features):
    """
    读取单个 edge 数据文件，返回：
       times: (T,)
       edges: list[str]
       X_tef: (T, E, F)
    """
    print(f"[INFO] reading XML: {path_xml}")
    tree = ET.parse(path_xml)
    root = tree.getroot()

    intervals = list(root.iter("interval"))
    print(f"[INFO] found {len(intervals)} <interval> in {path_xml}")
    if len(intervals) == 0:
        return None

    edge_set = set()
    for iv in intervals:
        for e in iv.findall("edge"):
            edge_set.add(e.attrib.get("id"))

    edges = sorted(edge_set)
    e_index = {eid: i for i, eid in enumerate(edges)}
    E, F = len(edges), len(features)

    times = []
    X = np.zeros((len(intervals), E, F), dtype=np.float32)

    for t, iv in enumerate(intervals):
        begin = float(iv.attrib.get("begin", "0"))
        end = float(iv.attrib.get("end", "0"))
        times.append(0.5 * (begin + end) if end > begin else begin)

        for e in iv.findall("edge"):
            eid = e.attrib.get("id")
            if eid not in e_index:
                continue
            i = e_index[eid]
            row = []
            for fname in features:
                v = e.attrib.get(fname, "0")
                try:
                    row.append(float(v))
                except Exception:
                    row.append(0.0)
            X[t, i, :] = row

    return np.array(times, dtype=np.float32), edges, X


def add_spatial_features(X_tef, edges, features, topo, spatial_features):
    """
    为数据添加上下游聚合特征
    
    Args:
        X_tef: (T, E, F) - 原始特征
        edges: list[str] - 边 ID
        features: list[str] - 特征名
        topo: NetworkTopology - 网络拓扑
        spatial_features: list[str] - 要聚合的特征名
    
    Returns:
        X_aug: (T, E, F+2*len(spatial_features)) - 增强后的特征
        feat_names_aug: list[str] - 新的特征名列表
    """
    if topo is None or not spatial_features:
        return X_tef, features
    
    print(f"[INFO] Adding spatial features: {spatial_features}")
    
    T, E, F = X_tef.shape
    edge_to_idx = {eid: i for i, eid in enumerate(edges)}
    
    # 为每个空间特征和聚合类型创建数据
    spatial_arrays = []
    feat_names_aug = list(features)
    
    for feat_name in spatial_features:
        if feat_name not in features:
            print(f"[WARN] {feat_name} not in features, skipping")
            continue
        
        feat_idx = features.index(feat_name)
        
        # 上游均值
        upstream_mean = np.zeros((T, E), dtype=np.float32)
        # 下游均值
        downstream_mean = np.zeros((T, E), dtype=np.float32)
        
        for e_idx, edge_id in enumerate(edges):
            upstream_ids, downstream_ids = topo.get_neighbors(edge_id)
            
            # 过滤存在的邻居
            upstream_ids = [e for e in upstream_ids if e in edge_to_idx]
            downstream_ids = [e for e in downstream_ids if e in edge_to_idx]
            
            # 上游聚合
            if upstream_ids:
                upstream_indices = [edge_to_idx[e] for e in upstream_ids]
                upstream_mean[:, e_idx] = X_tef[:, upstream_indices, feat_idx].mean(axis=1)
            
            # 下游聚合
            if downstream_ids:
                downstream_indices = [edge_to_idx[e] for e in downstream_ids]
                downstream_mean[:, e_idx] = X_tef[:, downstream_indices, feat_idx].mean(axis=1)
        
        spatial_arrays.append(upstream_mean[..., None])  # (T, E, 1)
        spatial_arrays.append(downstream_mean[..., None])  # (T, E, 1)
        
        feat_names_aug.append(f"{feat_name}_upstream_mean")
        feat_names_aug.append(f"{feat_name}_downstream_mean")
    
    # 拼接
    X_aug = np.concatenate([X_tef] + spatial_arrays, axis=2)
    
    print(f"[INFO] Feature dimensions: {F} -> {len(feat_names_aug)}")
    
    return X_aug, feat_names_aug


def build_one_scenario(
    sdir: Path,
    features,
    lag_features,
    lags,
    target,
    horizon,
    topo=None,
    add_spatial=False,
    spatial_features=None,
):
    """构建单个场景的数据"""
    candidates = ["edgedata.xml", "edgedump.add.xml"]
    x_path = None
    for name in candidates:
        p = sdir / name
        if p.exists():
            x_path = p
            break

    if x_path is None:
        print(f"[WARN] {sdir} has no edgedata.xml or edgedump.add.xml")
        return None

    parsed = read_edgedata_xml(str(x_path), features)
    if parsed is None:
        print(f"[WARN] {sdir} has no valid <interval> in {x_path.name}")
        return None

    times, edges, X_tef = parsed  # (T, E, F)
    
    # ===== 关键改进：过滤掉全 0 的时间步 =====
    # 原因：仿真从 0 秒开始记录，但车辆在 ~8 小时后才开始出现
    # 前面的时间步全是 0（没有流量），会干扰学习
    zero_mask = np.sum(X_tef, axis=(1, 2)) == 0
    if np.any(zero_mask):
        first_valid = np.where(~zero_mask)[0]
        if len(first_valid) > 0:
            first_valid_idx = first_valid[0]
            skipped_time = times[first_valid_idx]
            print(f"[INFO] {sdir.name}: 丢弃前 {first_valid_idx} 个全 0 时间步 "
                  f"({skipped_time/3600:.2f}h，第 {skipped_time:.0f} 秒)")
            X_tef = X_tef[first_valid_idx:]
            times = times[first_valid_idx:]
        else:
            print(f"[WARN] {sdir.name}: 所有数据都是 0！")
            return None
    
    # 添加空间特征
    if add_spatial and topo and spatial_features:
        X_tef, features_aug = add_spatial_features(
            X_tef, edges, list(features), topo, spatial_features
        )
    else:
        features_aug = list(features)
    
    T, E, F = X_tef.shape

    # 目标特征
    try:
        target_idx = features_aug.index(target)
    except ValueError:
        raise RuntimeError(f"target '{target}' not in features")

    # 生成滞后特征
    lag_names = []
    lag_arrays = []
    lag_feat_idx = [features_aug.index(f) for f in lag_features if f in features_aug]
    for fidx, fname in zip(lag_feat_idx, [f for f in lag_features if f in features_aug]):
        base = X_tef[:, :, fidx]  # (T, E)
        for k in lags:
            roll = np.roll(base, k, axis=0)
            roll[:k, :] = np.nan
            lag_arrays.append(roll[..., None])
            lag_names.append(f"{fname}_lag{k}")

    if lag_arrays:
        L = np.concatenate(lag_arrays, axis=2)
        X_all = np.concatenate([X_tef, L], axis=2)
        feat_names = features_aug + lag_names
    else:
        X_all = X_tef
        feat_names = features_aug

    max_lag = max(lags) if lags else 0
    valid_t_start = max_lag
    valid_t_end = T - horizon
    if valid_t_end <= valid_t_start:
        return None

    Xv = X_all[valid_t_start:valid_t_end, :, :]
    yv = X_tef[valid_t_start + horizon : valid_t_end + horizon, :, target_idx]
    tv = times[valid_t_start:valid_t_end]

    Tv, E, D = Xv.shape
    X_flat = Xv.reshape(Tv * E, D).astype(np.float32)
    Y_flat = yv.reshape(Tv * E, 1).astype(np.float32)

    meta = {
        "scenario": sdir.name,
        "T": int(T),
        "Tv": int(Tv),
        "E": int(E),
        "features": feat_names,
        "target": target,
        "horizon": horizon,
        "lags": lags,
        "lag_features": lag_features,
        "add_spatial": add_spatial,
        "spatial_features": spatial_features if spatial_features else [],
        "interval_times": tv[:10].tolist(),
    }
    return X_flat, Y_flat, feat_names, meta, edges


def main():
    args = parse_args()
    sroot = Path(args.scenarios_dir)
    
    # 加载网络拓扑
    topo = None
    add_spatial = args.add_spatial.lower() == "true"
    if add_spatial:
        if NetworkTopology is None:
            print("[ERROR] NetworkTopology not available, cannot add spatial features")
            return
        try:
            topo = NetworkTopology(args.network_file)
            print(f"[OK] Loaded network topology from {args.network_file}")
        except Exception as e:
            print(f"[ERROR] Failed to load network: {e}")
            add_spatial = False
    
    dirs = sorted([d for d in sroot.iterdir() if d.is_dir() and d.name.upper().startswith("S")])

    X_list, Y_list, metas, edges_map = [], [], [], {}
    for d in dirs:
        r = build_one_scenario(
            d,
            args.features,
            args.lag_features,
            args.lags,
            args.target,
            args.horizon,
            topo=topo,
            add_spatial=add_spatial,
            spatial_features=args.spatial_features if add_spatial else None,
        )
        if r is None:
            print(f"[SKIP] {d} (no edgedata or too few intervals)")
            continue
        Xf, Yf, feat_names, meta, edges = r
        X_list.append(Xf)
        Y_list.append(Yf)
        metas.append(meta)
        edges_map[meta["scenario"]] = edges
        print(
            f"[OK] {d.name}: Tv={meta['Tv']}, E={meta['E']}, "
            f"D={len(feat_names)}, samples={len(Xf)}"
        )

    if not X_list:
        print("No data collected.")
        return

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_tr = int(n * 0.8)
    n_va = int(n * 0.1)
    split = {
        "train": idx[:n_tr],
        "val": idx[n_tr : n_tr + n_va],
        "test": idx[n_tr + n_va :],
    }

    meta_all = {
        "features": feat_names,
        "target": args.target,
        "horizon": args.horizon,
        "lags": args.lags,
        "lag_features": args.lag_features,
        "add_spatial": add_spatial,
        "spatial_features": args.spatial_features if add_spatial else [],
        "scenarios": [m["scenario"] for m in metas],
        "per_scenario": metas,
        "edges": edges_map,
    }

    np.savez_compressed(
        args.out_npz,
        X=X,
        Y=Y,
        features=np.array(feat_names, dtype=object),
        target=np.array(args.target, dtype=object),
        meta=np.array(meta_all, dtype=object),
        split=np.array(split, dtype=object),
    )
    print(f"[DONE] saved: {args.out_npz}  X={X.shape} Y={Y.shape}")

    if args.write_csv_preview:
        m = min(5000, len(X))
        df = pd.DataFrame(X[:m], columns=feat_names)
        df["y_next_" + args.target] = Y[:m, 0]
        df.to_csv(args.write_csv_preview, index=False)
        print(f"[INFO] preview csv: {args.write_csv_preview} (rows={m})")


if __name__ == "__main__":
    main()
