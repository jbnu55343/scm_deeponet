# -*- coding: utf-8 -*-
# 用法：
#   python snap_c101_to_shanghai.py
#
# 输出：
#   1) customers_snapped.csv   （包含吸附后的 x,y、edge、lane、pos 等）
#   2) customers_snapped.poi.xml（可在 netedit 中加载的 POI，可视化检查）

import os
import math
import csv
import xml.etree.ElementTree as ET

# === 需要你改的部分：文件路径 ===
NET_PATH = r"D:\\pro_and_data\SCM_DeepONet\shanghai_center.net.xml"       # <<< 改成你的 net.xml
C101_PATH = r"D:\\pro_and_data\SCM_DeepONet\solomon_dataset\\C1\\C101.csv"     # <<< 改成你的 C101.csv
OUT_CSV   = "customers_snapped.csv"
OUT_POI   = "customers_snapped.poi.xml"

# === 需要你确认/改的部分：列名 ===
COL_ID   = "CUST NO."           # 客户索引列名
COL_X    = "XCOORD."            # C101 x 列名
COL_Y    = "YCOORD."            # C101 y 列名
COL_DEM  = "DEMAND"       # 需求
COL_RT   = "READY TIME"   # 准备时间
COL_DD   = "DUE DATE"     # 到期日
COL_ST   = "SERVICE TIME" # 服务时间

# === 可调参数 ===
PADDING_RATIO = 0.05     # 把点嵌入到路网边界内侧留 5% 边距
SEARCH_RADII  = [50, 100, 200, 400, 800]  # 逐步扩大找最近道路的半径（米）
SKIP_INTERNAL = True      # 跳过内部边（internal edges）

# ---------- 依赖 sumolib ----------
try:
    import sumolib
    from sumolib.geomhelper import distancePointToLine
except Exception as e:
    raise SystemExit("请先安装 SUMO 并确保 python 能 import sumolib。错误：%s" % e)

# ---------- 读取 C101 ----------
def read_c101(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            # 统一字段名读取
            rid = str(row[COL_ID]).strip()
            x   = float(row[COL_X])
            y   = float(row[COL_Y])
            demand      = float(row.get(COL_DEM, 0) or 0)
            ready_time  = float(row.get(COL_RT, 0) or 0)
            due_date    = float(row.get(COL_DD, 0) or 0)
            service_time= float(row.get(COL_ST, 0) or 0)
            rows.append({
                "id": rid, "x": x, "y": y,
                "demand": demand,
                "ready_time": ready_time,
                "due_date": due_date,
                "service_time": service_time
            })
    if not rows:
        raise RuntimeError("C101.csv 读到了 0 行。请检查路径和编码/分隔符。")
    return rows

# ---------- 嵌入坐标到 convBoundary（等比例缩放+平移；自动选择是否旋转90°以更好贴合） ----------
def get_conv_boundary_from_location(net_path):
    # 直接从 <location convBoundary="..."> 读
    root = ET.parse(net_path).getroot()
    loc = root.find("location")
    if loc is None or ("convBoundary" not in loc.attrib):
        return None
    vals = list(map(float, loc.attrib["convBoundary"].split(",")))
    if len(vals) != 4:
        return None
    return tuple(vals)  # (minx, miny, maxx, maxy)

def compute_affine_embed(points, bbox, padding_ratio=0.05):
    """
    points: [{'x':..,'y':..}, ...]  in C101 coords
    bbox  : (minx, miny, maxx, maxy)  SUMO convBoundary
    返回：transform(dict)，包含：
      rot (0 或 90)，scale，tx，ty
      以及一个函数 transform_xy(x,y) -> (X,Y)
    """
    minx = min(p["x"] for p in points); maxx = max(p["x"] for p in points)
    miny = min(p["y"] for p in points); maxy = max(p["y"] for p in points)
    cw, ch = maxx - minx, maxy - miny
    if cw <= 0 or ch <= 0:
        raise RuntimeError("C101 坐标范围异常")

    bx0, by0, bx1, by1 = bbox
    bw, bh = bx1 - bx0, by1 - by0
    padw, padh = bw * padding_ratio, bh * padding_ratio
    inner_w, inner_h = bw - 2*padw, bh - 2*padh

    def fit(rot90):
        # 旋转前： (x,y) -> (x,y)；旋转后： (x,y)->(y, -x) 或者简单交换轴；这里用 90°旋转并保持 y 正向
        if rot90:
            w, h = ch, cw
        else:
            w, h = cw, ch
        scale = min(inner_w / w, inner_h / h)
        # 居中放置
        if rot90:
            # 嵌入后 X = bx0+padw + ( (y - miny)*scale )
            #        Y = by0+padh + ( ( (maxx - x) )*scale )   （让整体不倒置，效果更直观）
            tx = bx0 + padw - miny * scale
            ty = by0 + padh + maxx * scale
            return scale, tx, ty
        else:
            # X = bx0+padw + (x - minx)*scale
            # Y = by0+padh + (y - miny)*scale
            tx = bx0 + padw - minx * scale
            ty = by0 + padh - miny * scale
            return scale, tx, ty

    # 试 0° 与 90°，选能放得更大的那个（scale 更大）
    s0, tx0, ty0 = fit(False)
    s1, tx1, ty1 = fit(True)
    if s1 > s0:
        rot, scale, tx, ty = 90, s1, tx1, ty1
    else:
        rot, scale, tx, ty = 0, s0, tx0, ty0

    def transform_xy(x, y):
        if rot == 90:
            X = y * scale + tx
            Y = (maxx - x) * scale + ty
        else:
            X = x * scale + tx
            Y = y * scale + ty
        return X, Y

    return {"rot": rot, "scale": scale, "tx": tx, "ty": ty, "transform": transform_xy}

# ---------- 几何：投影点到折线，返回距离与沿线位置 pos ----------
def project_point_to_polyline(px, py, poly):
    """
    poly: [(x0,y0), (x1,y1), ...]
    返回：(minDist, proj_x, proj_y, pos_along)
      pos_along 为从 poly[0] 起沿折线的长度（米）
    """
    best = (float("inf"), None, None, None)
    acc_len = 0.0
    for i in range(len(poly)-1):
        x1, y1 = poly[i]
        x2, y2 = poly[i+1]
        vx, vy = x2 - x1, y2 - y1
        seg_len2 = vx*vx + vy*vy
        if seg_len2 == 0:
            # 退化为点
            dx, dy = px - x1, py - y1
            d = math.hypot(dx, dy)
            cand = (d, x1, y1, acc_len)
        else:
            # 投影参数 t
            t = ((px - x1)*vx + (py - y1)*vy) / seg_len2
            t = max(0.0, min(1.0, t))
            qx, qy = x1 + t*vx, y1 + t*vy
            d = math.hypot(px - qx, py - qy)
            cand = (d, qx, qy, acc_len + math.sqrt(seg_len2)*t)
        if cand[0] < best[0]:
            best = cand
        acc_len += math.sqrt(seg_len2)
    return best  # (dist, qx, qy, pos)

# ---------- 吸附到最近道路 ----------
def snap_to_network(net, x, y, radii, skip_internal=True):
    for r in radii:
        candidates = net.getNeighboringEdges(x, y, r)
        # candidates: [(edge, distance), ...]
        # 过滤 internal / 特殊边
        cand2 = []
        for e, _ in candidates:
            if skip_internal and e.getFunction() == "internal":
                continue
            cand2.append(e)
        if not cand2:
            continue

        best = (float("inf"), None, None, None, None)  # dist, edge, lane, qx, qy, pos
        for e in cand2:
            for ln in e.getLanes():
                shape = ln.getShape()  # polyline
                dist, qx, qy, pos = project_point_to_polyline(x, y, shape)
                if dist < best[0]:
                    best = (dist, e, ln, qx, qy, pos)
        if best[1] is not None:
            dist, e, ln, qx, qy, pos = best
            return {
                "edge_id": e.getID(),
                "lane_id": ln.getID(),
                "lane_len": ln.getLength(),
                "x_snap": qx,
                "y_snap": qy,
                "pos": pos,
                "dist": dist
            }
    return None

# ---------- 主流程 ----------
def main():
    print("[1/4] 读取路网 …")
    net = sumolib.net.readNet(NET_PATH)
    bbox = get_conv_boundary_from_location(NET_PATH)
    if bbox is None:
        # 兜底：用边的几何求边界
        print("未从 <location> 读到 convBoundary，使用路网几何计算边界")
        xs, ys = [], []
        for e in net.getEdges():
            for ln in e.getLanes():
                xs += [p[0] for p in ln.getShape()]
                ys += [p[1] for p in ln.getShape()]
        bbox = (min(xs), min(ys), max(xs), max(ys))

    print("[2/4] 读取 C101 数据 …")
    pts = read_c101(C101_PATH)
    print("C101 点数：", len(pts))

    print("[3/4] 计算嵌入变换并生成内部坐标 …")
    tfm = compute_affine_embed(pts, bbox, PADDING_RATIO)
    rot, scale, tx, ty = tfm["rot"], tfm["scale"], tfm["tx"], tfm["ty"]
    transform_xy = tfm["transform"]
    for p in pts:
        X, Y = transform_xy(p["x"], p["y"])
        p["x_sumo"] = X
        p["y_sumo"] = Y

    print(f"嵌入参数：rot={rot}°, scale={scale:.4f}, tx={tx:.2f}, ty={ty:.2f}")

    print("[4/4] 吸附到最近道路 …")
    out_rows = []
    miss = 0
    for p in pts:
        res = snap_to_network(net, p["x_sumo"], p["y_sumo"], SEARCH_RADII, SKIP_INTERNAL)
        if res is None:
            miss += 1
            continue
        out_rows.append({
            "id": p["id"],
            "x_src": p["x"],
            "y_src": p["y"],
            "x_sumo": p["x_sumo"],
            "y_sumo": p["y_sumo"],
            "edge": res["edge_id"],
            "lane": res["lane_id"],
            "pos": round(res["pos"], 3),
            "snap_x": round(res["x_snap"], 3),
            "snap_y": round(res["y_snap"], 3),
            "snap_dist": round(res["dist"], 3),
            "demand": p["demand"],
            "ready_time": p["ready_time"],
            "due_date": p["due_date"],
            "service_time": p["service_time"]
        })

    # 写 CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # 写 POI（吸附后的点）
    pois = ET.Element("additional")
    for r in out_rows:
        poi = ET.SubElement(pois, "poi", {
            "id": f"c_{r['id']}",
            "x": str(r["snap_x"]),
            "y": str(r["snap_y"]),
            "color": "1,0,0",
            "layer": "10"
        })
    ET.ElementTree(pois).write(OUT_POI, encoding="utf-8", xml_declaration=True)

    print(f"完成：成功吸附 {len(out_rows)} 个点，失败 {miss} 个。")
    print(f"- {OUT_CSV}")
    print(f"- {OUT_POI}")
    if miss > 0:
        print("提示：有未吸附的点，考虑把 SEARCH_RADII 调大，或核查路网范围/嵌入参数。")

if __name__ == "__main__":
    main()
