r"""
Usage

PowerShell:
```powershell
python scripts\resnap_customers_v2.py `
  --net net\shanghai_5km.net.xml `
  --in_csv data\solomon\C1\C101.csv `
  --out_csv data\customers_snapped_5km.csv `
  --map fit --fit_margin 150 `
  --type_prefix "highway.primary,highway.secondary,highway.tertiary,highway.residential,highway.service,highway.living_street,highway.unclassified" `
  --max_radius 2500 `
  --grow 1.6
"""


# -*- coding: utf-8 -*-
# resnap_customers.py —— 一键重跑吸附：预设 r120 / r200，导出已吸附+未吸附（加入坐标映射 + 自适应半径）
from __future__ import annotations
import os, sys, csv, math, argparse
from typing import List, Tuple, Optional
from collections import Counter

# 1) SUMO 环境
if "SUMO_HOME" not in os.environ:
    print("ERROR: 请先设置 SUMO_HOME，例如：$env:SUMO_HOME='D:\\software2'")
    sys.exit(1)
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
from sumolib.net import readNet

# 2) 你的常用预设（改这里就行）
PRESETS = {
    "r120": {
        "net":       r"F:\git_local\scm_deeponet\shanghai.net.xml",
        "in_csv":    r"F:\git_local\scm_deeponet\solomon_dataset\C1\C103.csv",
        "out_csv":   r"F:\git_local\scm_deeponet\customers_snapped_new.csv",
        "radius":    120,
        # 限定居民/支路；需要更宽松就切 r200
        "type_prefix": "highway.residential,highway.service,highway.living_street,highway.unclassified",
    },
    "r200": {
        "net":       r"F:\git_local\scm_deeponet\shanghai.net.xml",
        "in_csv":    r"F:\git_local\scm_deeponet\solomon_dataset\C1\C103.csv",
        "out_csv":   r"F:\git_local\scm_deeponet\customers_snapped_r200.csv",
        "radius":    200,
        "type_prefix": "",
    },
}

# ---------- 工具函数 ----------
def ensure_outdir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def build_parser():
    p = argparse.ArgumentParser("Snap customers to SUMO net (带预设 + 坐标映射 + 自适应半径)")
    p.add_argument("--preset", choices=PRESETS.keys(), default="r120")
    p.add_argument("--net")
    p.add_argument("--in_csv")
    p.add_argument("--out_csv")
    p.add_argument("--radius", type=float)
    p.add_argument("--type_prefix", default=None, help="逗号分隔；空串\"\"=不限制")

    # 映射相关
    p.add_argument("--map", choices=["none", "linear", "fit"], default="none")
    p.add_argument("--swap_xy", action="store_true", help="交换 CSV 的 X/Y")
    p.add_argument("--scale_x", type=float, default=None)
    p.add_argument("--scale_y", type=float, default=None)
    p.add_argument("--offset_x", type=float, default=0.0)
    p.add_argument("--offset_y", type=float, default=0.0)
    p.add_argument("--rotate_deg", type=float, default=0.0, help="绕 CSV 原点逆时针旋转")
    p.add_argument("--roi", type=str, default=None, help="手动 ROI: xmin,ymin,xmax,ymax（米）")
    p.add_argument("--fit_margin", type=float, default=0.0, help="fit 模式内缩边距（米）")

    # 自适应半径
    p.add_argument("--max_radius", type=float, default=3000.0, help="找不到时扩大到的最大半径")
    p.add_argument("--grow", type=float, default=1.6, help="半径扩大倍率（>1）")

    p.add_argument("--dryrun", action="store_true", help="只做映射预览，不执行吸附")
    return p

def merge_args(preset_name, args):
    cfg = PRESETS[preset_name].copy()
    for k in ["net","in_csv","out_csv","radius","type_prefix"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if cfg["type_prefix"] is None:
        cfg["type_prefix"] = ""
    cfg["type_prefix"] = [s.strip() for s in cfg["type_prefix"].split(",") if s.strip()!=""]
    return cfg

def edge_type_allowed(edge, allowed_prefixes: List[str]) -> bool:
    # 一律屏蔽功能性内部/非车行边
    func = edge.getFunction() or ""
    if func in {"internal", "connector", "walkingarea", "railway"}:
        return False
    if not allowed_prefixes:
        return True
    etype = edge.getType() or ""
    for pre in allowed_prefixes:
        if etype == pre or etype.startswith(pre):
            return True
    # 若路网未设置 type，则默认放行，避免过严
    return etype == ""

def point_in_bbox(x, y, bbox) -> bool:
    (minx, miny), (maxx, maxy) = bbox
    return (minx - 1e-6) <= x <= (maxx + 1e-6) and (miny - 1e-6) <= y <= (maxy + 1e-6)

def proj_point_to_polyline(px, py, poly: List[Tuple[float,float]]):
    best = (None, None, float("inf"), 0.0)
    cum = 0.0
    for i in range(len(poly)-1):
        x1,y1 = poly[i]; x2,y2 = poly[i+1]
        vx,vy = (x2-x1, y2-y1)
        wx,wy = (px-x1, py-y1)
        seg_len2 = vx*vx + vy*vy
        if seg_len2 == 0:
            t = 0.0
        else:
            t = max(0.0, min(1.0, (wx*vx + wy*vy)/seg_len2))
        projx, projy = (x1 + t*vx, y1 + t*vy)
        dx,dy = (px-projx, py-projy)
        dist = math.hypot(dx,dy)
        if dist < best[2]:
            best = (projx, projy, dist, cum + t*math.sqrt(seg_len2))
        cum += math.sqrt(seg_len2)
    return best[3], (best[0], best[1]), best[2]

def find_snap_on_net(net, x, y, radius, allowed_types: List[str], max_radius=800.0, grow=1.6):
    """自适应扩大半径：从 radius 开始，逐步扩大到 max_radius，直到找到最佳匹配"""
    r = max(1e-6, radius)
    best = None
    while r <= max_radius and best is None:
        neigh = net.getNeighboringEdges(x, y, r)
        if not neigh:
            r *= grow
            continue
        candidates = []
        for edge, dist in neigh:
            if edge_type_allowed(edge, allowed_types):
                candidates.append((edge, dist))
        if not candidates:
            r *= grow
            continue
        # 在候选边的车道上做几何投影，取距离最近者
        for edge, _ in candidates:
            for lane in edge.getLanes():
                shape = lane.getShape(includeJunctions=True)
                s, _, d = proj_point_to_polyline(x, y, shape)
                if best is None or d < best["dist"]:
                    best = {"edge": edge, "lane": lane, "lanePos": s, "dist": d}
        if best is None:
            r *= grow
    return best

def read_points(csv_path: str):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    def pick(row, cands):
        for c in cands:
            if c in row:
                return c
        return None
    if not rows:
        return [], None, None
    sample = rows[0]
    id_col = pick(sample, ["CUST NO.","CUST_NO","CUST_ID","ID","CustNo"])
    x_col  = pick(sample, ["XCOORD.","XCOORD","X","X_COORD"])
    y_col  = pick(sample, ["YCOORD.","YCOORD","Y","Y_COORD"])
    if not (id_col and x_col and y_col):
        raise RuntimeError(f"找不到必要列名，检测到列：{list(sample.keys())}")
    # 统计 CSV bbox
    xs, ys = [], []
    for r in rows:
        try:
            xs.append(float(r[x_col])); ys.append(float(r[y_col]))
        except Exception:
            pass
    csv_bbox = ( (min(xs), min(ys)), (max(xs), max(ys)) ) if xs and ys else None
    return rows, (id_col, x_col, y_col), csv_bbox

# ---------- 映射模块 ----------
def build_mapper(args, csv_bbox, net_bbox):
    # 解析 ROI
    if args.roi:
        parts = [float(s) for s in args.roi.split(",")]
        if len(parts)!=4: raise ValueError("--roi 需为 xmin,ymin,xmax,ymax")
        roi = ((parts[0], parts[1]), (parts[2], parts[3]))
    else:
        roi = net_bbox

    def linear_params():
        sx = args.scale_x if args.scale_x is not None else 1.0
        sy = args.scale_y if args.scale_y is not None else sx
        th = math.radians(args.rotate_deg or 0.0)
        c, s = math.cos(th), math.sin(th)
        def mapper(x, y):
            # 先可选 swap，再旋转（绕 CSV 原点），最后缩放、平移
            if args.swap_xy:
                x, y = y, x
            xr = c*x - s*y
            yr = s*x + c*y
            X = xr * sx + args.offset_x
            Y = yr * sy + args.offset_y
            return X, Y
        desc = f"linear: swap={args.swap_xy}, rot={args.rotate_deg}deg, scale=({sx},{sy}), offset=({args.offset_x},{args.offset_y})"
        return mapper, desc

    def fit_params():
        if csv_bbox is None:
            raise RuntimeError("fit 模式需要能解析出 CSV bbox")
        (cx0, cy0), (cx1, cy1) = csv_bbox
        csv_w, csv_h = (cx1-cx0, cy1-cy0)
        if csv_w<=0 or csv_h<=0:
            raise RuntimeError(f"CSV bbox 非法: {csv_bbox}")

        (rx0, ry0), (rx1, ry1) = roi
        rx0 += args.fit_margin; ry0 += args.fit_margin
        rx1 -= args.fit_margin; ry1 -= args.fit_margin
        roi_w, roi_h = (rx1-rx0, ry1-ry0)
        if roi_w<=0 or roi_h<=0:
            raise RuntimeError(f"ROI 太小或 fit_margin 过大: {roi}")

        s = min(roi_w/csv_w, roi_h/csv_h)  # 等比缩放
        # 把 CSV 的左下角平移到 ROI 的左下角，再加居中补偿
        X0 = rx0 - cx0*s
        Y0 = ry0 - cy0*s
        # 居中
        dx = (roi_w - csv_w*s)/2.0
        dy = (roi_h - csv_h*s)/2.0
        X0 += dx; Y0 += dy

        th = math.radians(args.rotate_deg or 0.0)
        c, sn = math.cos(th), math.sin(th)

        def mapper(x, y):
            # 可选 swap -> 旋转 -> 等比缩放+平移
            if args.swap_xy:
                x, y = y, x
            xr = c*x - sn*y
            yr = sn*x + c*y
            return xr*s + X0, yr*s + Y0
        desc = f"fit: s={round(s,3)}, offset=({round(X0,3)},{round(Y0,3)}), roi={roi}, margin={args.fit_margin}, rot={args.rotate_deg}deg, swap={args.swap_xy}"
        return mapper, desc

    if args.map == "none":
        def mapper(x,y):
            # 恒等映射；swap/rotate/scale 在 linear/fit 模式才生效
            return (x,y)
        return mapper, "none"
    elif args.map == "linear":
        return linear_params()
    elif args.map == "fit":
        return fit_params()
    else:
        raise ValueError("未知 map 模式")

def write_outputs(out_csv, snapped_rows, header_extra=None):
    fields = list(snapped_rows[0].keys()) if snapped_rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(snapped_rows)

# ---------- 主流程 ----------
def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = merge_args(args.preset, args)
    print("[INFO] 使用配置：", cfg)

    # 读取路网
    net = readNet(cfg["net"])
    bbox = net.getBBoxXY()
    (minx, miny), (maxx, maxy) = bbox
    print(f"[INFO] Net BBox: (({minx:.3f},{miny:.3f}),({maxx:.3f},{maxy:.3f}))  width={maxx-minx:.1f}m  height={maxy-miny:.1f}m")

    # 读点 & CSV bbox
    rows, cols, csv_bbox = read_points(cfg["in_csv"])
    id_col, x_col, y_col = cols
    if csv_bbox:
        (cx0,cy0),(cx1,cy1)=csv_bbox
        print(f"[INFO] CSV BBox: (({cx0:.3f},{cy0:.3f}),({cx1:.3f},{cy1:.3f}))  width={cx1-cx0:.3f}  height={cy1-cy0:.3f}  (CSV单位)")

    # 构建映射（统一入口；避免“双重 swap”）
    mapper, map_desc = build_mapper(args, csv_bbox, bbox)
    print(f"[INFO] 映射模式: {args.map}  （{map_desc}）")

    # 预览前 5 个点
    for i, row in enumerate(rows[:5], 1):
        try:
            x0 = float(row[x_col]); y0 = float(row[y_col])
            X, Y = mapper(x0, y0)
            print(f"  sample#{i}: CSV({x0:.3f},{y0:.3f}) -> XY({X:.3f},{Y:.3f}) in_bbox={point_in_bbox(X,Y,bbox)}")
        except Exception as e:
            print(f"  sample#{i}: 解析失败: {e}")

    if args.dryrun:
        print("[DRYRUN] 仅做映射预览，未执行吸附。")
        return

    snapped, unsnapped = [], []
    for row in rows:
        try:
            x = float(row[x_col]); y = float(row[y_col])
            X, Y = mapper(x, y)
        except Exception:
            unsnapped.append({**row, "SNAP_STATUS":"bad_xy"})
            continue

        if not point_in_bbox(X, Y, bbox):
            unsnapped.append({**row, "SNAP_STATUS":"out_of_bbox", "mapped_x":X, "mapped_y":Y})
            continue

        hit = find_snap_on_net(
            net, X, Y,
            cfg["radius"],
            cfg["type_prefix"],
            max_radius=args.max_radius,
            grow=args.grow
        )
        if hit is None:
            unsnapped.append({**row, "SNAP_STATUS":"no_edge_in_radius", "mapped_x":X, "mapped_y":Y})
            continue

        snapped.append({
            **row,
            "SNAP_STATUS": "ok",
            "mapped_x": round(X,3),
            "mapped_y": round(Y,3),
            "edge_id":  hit["edge"].getID(),
            "lane_id":  hit["lane"].getID(),
            "lane_pos": round(hit["lanePos"], 3),
            "snap_dist": round(hit["dist"], 3),
        })

    # 输出
    out_csv = cfg["out_csv"]
    base, ext = os.path.splitext(out_csv)
    unsnapped_csv = base + "_unsnapped" + ext

    if snapped:
        write_outputs(out_csv, snapped)
        print(f"[OK] 已吸附 {len(snapped)}/{len(rows)} → {out_csv}")
    else:
        print("[WARN] 没有任何点被吸附，请检查映射参数/半径/道路类型")

    if unsnapped:
        write_outputs(unsnapped_csv, unsnapped)
        print(f"[WARN] 未吸附 {len(unsnapped)}/{len(rows)} → {unsnapped_csv}")
        # 失败原因统计
        cnt = Counter([r["SNAP_STATUS"] for r in unsnapped])
        print("[STATS] Unsnapped reasons:", dict(cnt))

if __name__ == "__main__":
    main()



'''
python scripts\resnap_customers_v2.py `
  --net net\shanghai_5km.net.xml `
  --in_csv data\solomon\C1\C101.csv `
  --out_csv data\customers_snapped_5km.csv `
  --map fit --fit_margin 150 `
  --type_prefix "highway.primary,highway.secondary,highway.tertiary,highway.residential,highway.service,highway.living_street,highway.unclassified" `
  --max_radius 2500 `
  --grow 1.6
  '''
