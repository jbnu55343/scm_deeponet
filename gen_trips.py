# -*- coding: utf-8 -*-
"""
Standalone gen_trips.py
- 直接在本文件顶部修改 CONFIG 即可运行（无需命令行参数）
- 从 customers_snapped_*.csv 读取 edge_id（可按 DEMAND 加权抽样）
- 生成多个场景 S001, S002, ... ，每个场景一个 trips.trips.xml
"""

from __future__ import annotations
import os, random
from pathlib import Path
import pandas as pd

# ========= 配置（只改这里） =========
CONFIG = {
    # CSV 路径（可用绝对路径；相对路径是相对本 .py 文件）
    "CUST_CSV": r"customers_snapped_r200.csv",
    # 输出目录（会自动创建 S001/S002/...）
    "OUT_DIR":  r"scenarios",
    # 场景数量
    "SCENARIOS": 6,
    # 基础车辆数（每个场景会乘以 MULTIPLIERS[k] 做强度变化）
    "BASE_TRIPS": 5000,
    # 各场景的强度倍率（会循环取用）
    "MULTIPLIERS": [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
    # 各场景的出发时间窗（秒）（会循环取用），示例：早高峰/平峰/晚高峰
    "TIME_WINDOWS": [(7*3600, 9*3600), (8*3600, 10*3600), (17*3600, 19*3600)],
    # 是否过滤仓库/工厂（DEMAND==0）
    "EXCLUDE_ZERO_DEMAND": True,
    # 是否按需求量 DEMAND 加权抽样（更贴近真实：大需求更可能被抽中）
    "USE_DEMAND_WEIGHT": True,
    # 随机种子（保证可复现；每个场景会在此基础上 +场景号）
    "SEED": 42,
}
# ==================================

def _resolve_path(p: str) -> Path:
    """相对路径按当前脚本所在目录解析"""
    base = Path(__file__).resolve().parent
    return (base / p).resolve()

def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    """在 DataFrame 中按多种备选列名（不区分大小写）查找列"""
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None

def _load_edges_and_weights(csv_path: Path, exclude_zero: bool, use_weight: bool):
    df = pd.read_csv(csv_path)

    # 找列名
    edge_col = _find_col(df, ["edge_id", "edge", "EDGE_ID"])
    if edge_col is None:
        raise RuntimeError(f"未在 {csv_path} 找到 edge_id 列，请确认。实际列有：{list(df.columns)}")

    demand_col = _find_col(df, ["DEMAND", "demand", "Demand"])

    # 过滤无效 edge_id / 仓库
    df = df.dropna(subset=[edge_col]).copy()
    if exclude_zero and demand_col is not None:
        # DEMAND > 0 作为客户点
        df = df[pd.to_numeric(df[demand_col], errors="coerce").fillna(0) > 0]

    # 唯一边集合
    edges = df[edge_col].astype(str).unique().tolist()
    if len(edges) < 2:
        raise RuntimeError("有效客户边不足 2 条，无法生成 OD 对。请检查 CSV 或过滤条件。")

    # 权重
    if use_weight and demand_col is not None:
        g = df.groupby(edge_col)[demand_col].sum().astype(float)
        # 小心全 0 的情况
        weights = [max(g.get(e, 0.0), 0.0) for e in edges]
        if sum(weights) <= 0:
            # 退化到等权
            weights = None
    else:
        weights = None

    return edges, weights

def _choice(edges: list[str], weights: list[float] | None) -> str:
    if weights is None:
        return random.choice(edges)
    # Python 3.6+ 支持 weights
    return random.choices(edges, weights=weights, k=1)[0]

def _gen_one_trips(trips_path: Path, edges: list[str], weights, n_trips: int, t0: int, t1: int, sid: int):
    trips_path.parent.mkdir(parents=True, exist_ok=True)
    with trips_path.open("w", encoding="utf-8") as f:
        f.write("<trips>\n")
        for i in range(n_trips):
            # 起终点按权重独立采样；避免相同则重抽
            fr = _choice(edges, weights)
            to = _choice(edges, weights)
            # 允许最多尝试几次避免 from==to
            tries = 0
            while to == fr and tries < 5:
                to = _choice(edges, weights); tries += 1
            depart = random.randint(t0, t1)
            f.write(f'  <trip id="veh{sid:03d}_{i}" depart="{depart}" from="{fr}" to="{to}" type="passenger"/>\n')
        f.write("</trips>\n")

def main():
    cfg = CONFIG.copy()
    # 解析路径
    csv_path = _resolve_path(cfg["CUST_CSV"])
    out_dir  = _resolve_path(cfg["OUT_DIR"])

    # 载入边与权重
    edges, weights = _load_edges_and_weights(
        csv_path,
        exclude_zero=cfg["EXCLUDE_ZERO_DEMAND"],
        use_weight=cfg["USE_DEMAND_WEIGHT"],
    )
    print(f"[INFO] 候选边数: {len(edges)}（按需求加权={bool(weights)}）")
    print(f"[INFO] 输出目录: {out_dir}")

    # 生成场景
    S = int(cfg["SCENARIOS"])
    MULTS = cfg["MULTIPLIERS"]
    WINS  = cfg["TIME_WINDOWS"]

    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(1, S+1):
        # 每个场景独立 seed，保证可复现同时场景间有差异
        random.seed(cfg["SEED"] + k)

        sdir = out_dir / f"S{k:03d}"
        sdir.mkdir(parents=True, exist_ok=True)

        mult = MULTS[(k-1) % len(MULTS)]
        n_trips = max(1, int(cfg["BASE_TRIPS"] * mult))
        t0, t1 = WINS[(k-1) % len(WINS)]

        trips_path = sdir / "trips.trips.xml"
        _gen_one_trips(trips_path, edges, weights, n_trips, t0, t1, sid=k)
        print(f"[OK] {sdir.name}: trips={n_trips}, window=({t0},{t1}) → {trips_path.name}")

    print("\n完成。你现在可以在每个 Sxxx 目录下用 duarouter/sumo 继续后续流程。")
    # 如果你习惯双击运行 .py，窗口一闪而过，可以取消下面注释以便查看输出：
    # input("按回车键退出...")

if __name__ == "__main__":
    main()
