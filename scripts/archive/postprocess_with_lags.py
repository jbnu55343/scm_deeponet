# -*- coding: utf-8 -*-
"""
postprocess_with_lags.py
从 scenarios/*/edgedata.xml 重新构建数据集，并为选定特征添加时间滞后 (t-1, t-2, ...）
不需要重跑 SUMO，只读 XML。

用法示例：
python scripts/postprocess_with_lags.py \
  --scenarios_dir scenarios \
  --out_npz data/dataset_sumo_5km_lag12.npz \
  --features speed density occupancy waitingTime traveltime entered left \
  --lag_features speed density occupancy waitingTime traveltime \
  --lags 1 2 \
  --target speed \
  --horizon 1 \
  --write_csv_preview data/preview_samples_lag.csv
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        "Build dataset with time-lag features from edgedata.xml"
    )
    p.add_argument("--scenarios_dir", type=str, default="scenarios")
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="基础特征名（来自 edgedata.xml 的属性名）",
    )
    p.add_argument(
        "--lag_features",
        nargs="+",
        default=[],
        help="需要做滞后的特征子集（必须包含在 --features 里）",
    )
    p.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 2],
        help="时间滞后阶数，如 1 2 表示 t-1、t-2",
    )
    p.add_argument("--target", type=str, default="speed")
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--write_csv_preview", type=str, default="")
    return p.parse_args()


def read_edgedata_xml(path_xml: str, features):
    """
    读取单个 edge 数据文件（edgedata.xml 或 edgedump.add.xml），返回：
       times: (T,)  每个区间的时间戳
       edges: list[str]  边 ID 列表
       X_tef: (T, E, F)  三维数组
    """
    print(f"[INFO] reading XML: {path_xml}")
    tree = ET.parse(path_xml)
    root = tree.getroot()

    # 有些文件根标签不是 edgeData，我们用 iter 保守地找所有 interval
    intervals = list(root.iter("interval"))
    print(f"[INFO] found {len(intervals)} <interval> in {path_xml}")
    if len(intervals) == 0:
        return None

    # 收集所有 edge id
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



def build_one_scenario(sdir: Path, features, lag_features, lags, target, horizon):
    # 兼容两种文件名：老版 edgedata.xml 和你现在的 edgedump.add.xml
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

    times, edges, X_tef = parsed  # (T,E,F)
    
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
    
    T, E, F = X_tef.shape

    # 目标 y：用 target 特征的 t+horizon
    try:
        target_idx = features.index(target)
    except ValueError:
        raise RuntimeError(f"target '{target}' 必须在 --features 里")

    # 生成滞后特征
    lag_names = []
    lag_arrays = []  # list of (T,E,1)
    lag_feat_idx = [features.index(f) for f in lag_features]
    for fidx, fname in zip(lag_feat_idx, lag_features):
        base = X_tef[:, :, fidx]  # (T,E)
        for k in lags:
            roll = np.roll(base, k, axis=0)
            roll[:k, :] = np.nan  # 前 k 行无效
            lag_arrays.append(roll[..., None])  # (T,E,1)
            lag_names.append(f"{fname}_lag{k}")

    if lag_arrays:
        L = np.concatenate(lag_arrays, axis=2)  # (T,E,Ln)
        X_all = np.concatenate([X_tef, L], axis=2)  # (T,E,F+Ln)
        feat_names = features + lag_names
    else:
        X_all = X_tef
        feat_names = features[:]

    # 生成 y，并对齐（去掉前 max_lag 行与最后 horizon 行）
    max_lag = max(lags) if lags else 0
    valid_t_start = max_lag
    valid_t_end = T - horizon
    if valid_t_end <= valid_t_start:
        return None

    Xv = X_all[valid_t_start:valid_t_end, :, :]  # (Tv,E,D)
    yv = X_tef[
        valid_t_start + horizon : valid_t_end + horizon, :, target_idx
    ]  # (Tv,E)
    tv = times[valid_t_start:valid_t_end]

    # 展平为样本
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
        "interval_times": tv[:10].tolist(),  # 预览头 10 个
    }
    return X_flat, Y_flat, feat_names, meta, edges


def main():
    args = parse_args()
    sroot = Path(args.scenarios_dir)
    # 所有以 S 开头的子目录（S001–S006）
    dirs = sorted(
        [d for d in sroot.iterdir() if d.is_dir() and d.name.upper().startswith("S")]
    )

    X_list, Y_list, metas, edges_map = [], [], [], {}
    for d in dirs:
        r = build_one_scenario(
            d,
            args.features,
            args.lag_features,
            args.lags,
            args.target,
            args.horizon,
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

    # 简单划分（按样本随机打散）
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

    # 组装元数据
    meta_all = {
        "features": feat_names,
        "target": args.target,
        "horizon": args.horizon,
        "lags": args.lags,
        "lag_features": args.lag_features,
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

    # 预览 CSV（抽样 5000）
    if args.write_csv_preview:
        m = min(5000, len(X))
        df = pd.DataFrame(X[:m], columns=feat_names)
        df["y_next_" + args.target] = Y[:m, 0]
        df.to_csv(args.write_csv_preview, index=False)
        print(f"[INFO] preview csv: {args.write_csv_preview} (rows={m})")


if __name__ == "__main__":
    main()
