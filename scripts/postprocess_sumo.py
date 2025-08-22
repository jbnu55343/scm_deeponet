# -*- coding: utf-8 -*-
"""
postprocess_sumo.py
- 读取 scenarios/S001..S006 下的 edgedata.xml
- 解析成按时间片的 per-edge 指标矩阵
- 组装 (X_t -> Y_{t+1}) 监督样本，导出 npz / 也可导出一份检查用 csv

用法示例：
python3 scripts/postprocess_sumo.py \
  --scenarios_dir scenarios \
  --out_npz data/dataset_sumo_5km.npz \
  --features speed flow occupancy density traveltime waitingTime \
  --target speed \
  --horizon 1 \
  --write_csv_preview data/preview_samples.csv
"""
from __future__ import annotations
import os, argparse, glob, math, xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
import numpy as np
import csv

def parse_args():
    p = argparse.ArgumentParser("Build dataset from SUMO edgedata.xml")
    p.add_argument("--scenarios_dir", default="scenarios")
    p.add_argument("--pattern", default="S0*")  # S001..S006
    p.add_argument("--edgedata_name", default="edgedata.xml")
    p.add_argument("--features", nargs="+",
                   default=["speed","flow","occupancy","density","traveltime","waitingTime"])
    p.add_argument("--target", default="speed")
    p.add_argument("--horizon", type=int, default=1, help="预测步长：用 t 预测 t+horizon")
    p.add_argument("--min_intervals", type=int, default=2, help="至少要有这么多时间片才可做配对")
    p.add_argument("--out_npz", default="data/dataset_sumo_5km.npz")
    p.add_argument("--write_csv_preview", default=None)
    return p.parse_args()

def gfirst(elem, keys):
    for k in keys:
        v = elem.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                return float("nan")
    return float("nan")

def parse_edgedata_file(path):
    """
    返回：list of (begin,end, {edge_id: {metric: value}})
    """
    tree = ET.parse(path)
    root = tree.getroot()
    out = []
    for inter in root.iter("interval"):
        begin = float(inter.get("begin", inter.get("start", 0)))
        end   = float(inter.get("end",   inter.get("stop",  0)))
        data = {}
        for e in inter.findall("edge"):
            eid = e.get("id")
            if not eid:
                continue
            d = {}
            # 尽量兼容不同字段名
            d["speed"]       = gfirst(e, ["speed","meanSpeed","speedAbs"])
            d["flow"]        = gfirst(e, ["entered","flow"])
            d["occupancy"]   = gfirst(e, ["occupancy"])
            d["density"]     = gfirst(e, ["density"])
            d["traveltime"]  = gfirst(e, ["traveltime","timeLoss"])
            d["waitingTime"] = gfirst(e, ["waitingTime","waiting"])
            data[eid] = d
        out.append((begin, end, data))
    # 按时间排序
    out.sort(key=lambda t: (t[0], t[1]))
    return out

def main():
    args = parse_args()
    scen_dirs = sorted([d for d in glob.glob(os.path.join(args.scenarios_dir, args.pattern)) if os.path.isdir(d)])
    if not scen_dirs:
        raise SystemExit(f"No scenario dirs under {args.scenarios_dir} with pattern {args.pattern}")

    all_samples_X = []
    all_samples_Y = []
    meta_samples  = []  # (scenario, t_index, edge_id)

    for sdir in scen_dirs:
        edfile = os.path.join(sdir, args.edgedata_name)
        if not os.path.isfile(edfile):
            print(f"[WARN] missing edgedata: {edfile} (skip)")
            continue
        intervals = parse_edgedata_file(edfile)
        if len(intervals) < args.min_intervals:
            print(f"[WARN] {sdir} has only {len(intervals)} interval(s). skip.")
            continue

        # 统一边集合（取所有时间片出现过的边）
        edge_ids = sorted({eid for _,_,dat in intervals for eid in dat.keys()})
        edge_index = {eid:i for i,eid in enumerate(edge_ids)}
        T = len(intervals); E = len(edge_ids); F = len(args.features)

        # 构造 [T, E, F] & [T, E]（target）
        X = np.full((T, E, F), np.nan, dtype=np.float32)
        Y = np.full((T, E), np.nan, dtype=np.float32)

        # 填入
        for t,(begin,end,dat) in enumerate(intervals):
            for eid, d in dat.items():
                ei = edge_index[eid]
                # features
                feat_vals = []
                for k in args.features:
                    feat_vals.append(float(d.get(k, float("nan"))))
                X[t, ei, :] = np.array(feat_vals, dtype=np.float32)
                # target（同一时间片的 target 先放这里，后面做 horizon shift）
                Y[t, ei] = float(d.get(args.target, float("nan")))

        # 做 (t -> t+h) 配对
        h = args.horizon
        valid_T = T - h
        if valid_T <= 0:
            print(f"[WARN] {sdir}: T={T} < horizon={h}. skip.")
            continue

        X_t = X[:valid_T]            # [T-h, E, F]
        Y_tp = Y[h:]                 # [T-h, E]
        # 缺失填充（简单做法：NaN->0；也可改成插值/沿时间片前向填充）
        X_t = np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0)
        Y_tp = np.nan_to_num(Y_tp, nan=0.0, posinf=0.0, neginf=0.0)

        # 变成样本维度：[ (T-h)*E, F ] / [ (T-h)*E, 1 ]
        X_flat = X_t.reshape(-1, F)
        Y_flat = Y_tp.reshape(-1, 1)

        all_samples_X.append(X_flat)
        all_samples_Y.append(Y_flat)

        # 记录 meta：对应 (scenario, t_index, edge_id)
        for t in range(valid_T):
            for eid in edge_ids:
                meta_samples.append((os.path.basename(sdir), t, eid))

        print(f"[OK] {sdir}: T={T}, E={E}, F={F}, samples={X_flat.shape[0]}")

    if not all_samples_X:
        raise SystemExit("No samples collected. Check edgedata.xml content or intervals.")

    X = np.vstack(all_samples_X)
    Y = np.vstack(all_samples_Y)

    # 简单打乱并划分
    N = X.shape[0]
    idx = np.random.RandomState(42).permutation(N)
    X = X[idx]; Y = Y[idx]
    # 80/10/10
    n_tr = int(0.8*N); n_va = int(0.1*N)
    split = {
        "train": (0, n_tr),
        "valid": (n_tr, n_tr+n_va),
        "test":  (n_tr+n_va, N)
    }

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(args.out_npz,
                        X=X, Y=Y,
                        features=np.array(args.features, dtype=object),
                        target=np.array(args.target, dtype=object),
                        meta=np.array(meta_samples, dtype=object),
                        split=np.array([split["train"], split["valid"], split["test"]], dtype=object))
    print(f"[DONE] saved dataset: {args.out_npz}")
    print(f"       X shape: {X.shape}, Y shape: {Y.shape}")

    # 可选：导出一小份 CSV 预览
    if args.write_csv_preview:
        K = min(5000, X.shape[0])
        with open(args.write_csv_preview, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([*args.features, f"target_{args.target}"])
            for i in range(K):
                w.writerow([*X[i].tolist(), Y[i,0]])
        print(f"[INFO] preview csv: {args.write_csv_preview} (rows={K})")

if __name__ == "__main__":
    main()
