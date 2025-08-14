# -*- coding: utf-8 -*-
"""
build_net_from_osm.py  —  从 OSM 生成 SUMO 路网（兼容老版本 netconvert）

功能
- 支持两种区域指定方式：--center + --radius_km，或直接 --bbox
- 先用 SUMO 的 tools/osmGet.py 拉取 OSM（自动处理前缀/后缀、_bbox 命名等）
- 如 osmGet 失败，可选回退为直接请求 Overpass API
- 调用 netconvert 生成 .net.xml（自动探测支持的参数，老版本也能跑）
- 打印路网 BBox（米）用于快速自检

准备
- 已安装 SUMO，并设置环境变量 SUMO_HOME（包含 tools 与 data）
- netconvert 在 PATH（或位于 %SUMO_HOME%/bin）

用法示例（PowerShell）
python scripts\build_net_from_osm.py `
  --center 121.4737,31.2304 `
  --radius_km 5 `
  --out_osm net\shanghai_5km.osm.xml `
  --out_net net\shanghai_5km.net.xml `
  --overwrite
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# -------------------- 工具函数 --------------------
def check_sumo_env():
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        raise RuntimeError("未检测到环境变量 SUMO_HOME，请先设置（指向 SUMO 安装目录）。")
    osm_get = Path(sumo_home) / "tools" / "osmGet.py"
    if not osm_get.exists():
        raise RuntimeError(f"未找到 osmGet.py：{osm_get}")
    netconvert = shutil.which("netconvert") or shutil.which(str(Path(sumo_home) / "bin" / "netconvert"))
    if not netconvert:
        raise RuntimeError("未找到 netconvert，可将 SUMO 的 bin 目录加入 PATH。")
    return str(osm_get), str(netconvert), Path(sumo_home)


def run(cmd: list[str], cwd: Optional[str] = None, check: bool = True) -> int:
    print("\n[CMD]", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, cwd=cwd)
    if check and p.returncode != 0:
        raise RuntimeError(f"命令执行失败（返回码 {p.returncode}）")
    return p.returncode


def get_help_text(exe: str) -> str:
    r = subprocess.run([exe, "--help"], capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def supports(help_text: str, opt: str) -> bool:
    return opt in help_text


def lonlat_bbox_from_center(center_lon: float, center_lat: float, radius_km: float):
    # 1°纬度 ≈ 110.574 km；1°经度 ≈ 111.320*cos(lat) km
    dlat = radius_km / 110.574
    dlon = radius_km / (111.320 * math.cos(math.radians(center_lat)))
    return (center_lon - dlon, center_lat - dlat, center_lon + dlon, center_lat + dlat)


def normalize_prefix_for_osm(out_osm: Path) -> Path:
    """
    osmGet.py 的 -p 参数是“前缀 prefix”，会自动追加 .osm.xml。
    用户如果传了 xxx.osm.xml，这里要把两层后缀去掉，得到标准前缀。
    """
    prefix = out_osm
    # 去掉 ".xml" 后再判断是否是 ".osm"
    if prefix.suffix == ".xml" and prefix.with_suffix("").suffix == ".osm":
        prefix = prefix.with_suffix("").with_suffix("")  # 连去两层后缀
    return prefix


def find_downloaded_osm(prefix: Path) -> Optional[Path]:
    """
    兼容多种实际落盘名：
      - <prefix>.osm.xml（常见）
      - <prefix>_bbox.osm.xml（osmGet 新版常见）
      - <prefix>.osm（少见）
      - <prefix>.osm.xml_bbox.osm.xml（用户误传 -p 为完整文件名导致的双后缀）
    返回找到的非空文件中“最近修改”的一个。
    """
    parent = prefix.parent
    candidates = []
    # 精确候选
    candidates += [
        parent / f"{prefix.name}.osm.xml",
        parent / f"{prefix.name}_bbox.osm.xml",
        parent / f"{prefix.name}.osm",
        parent / f"{prefix.name}.osm.xml_bbox.osm.xml",
    ]
    # 宽泛模式：凡是前缀匹配且包含 ".osm"
    for p in parent.glob(prefix.name + "*"):
        if ".osm" in p.name:
            candidates.append(p)

    valid = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    if not valid:
        return None
    # 选最近修改的
    valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return valid[0]


def download_osm_with_osmget(osm_get: str, bbox_str: str, out_osm: Path) -> Path:
    """使用 osmGet.py 下载，并将结果规范化为 out_osm 路径"""
    prefix = normalize_prefix_for_osm(out_osm)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] osmGet 输出前缀：{prefix}  （实际文件通常为 {prefix}.osm.xml 或 {prefix}_bbox.osm.xml）")
    run([sys.executable, osm_get, "-v", "-b", bbox_str, "-p", str(prefix)], check=True)

    actual = find_downloaded_osm(prefix)
    if not actual:
        raise RuntimeError("osmGet 未生成有效的 .osm(.xml) 文件（可能网络失败/被限流）。")

    # 归一化到 out_osm
    if actual.resolve() != out_osm.resolve():
        out_osm.parent.mkdir(parents=True, exist_ok=True)
        # Windows 上 replace 可跨卷移动；如不行可改为 copy+unlink
        out_osm.unlink(missing_ok=True)
        actual.replace(out_osm)
    print(f"[OK] 已获取 OSM：{out_osm}  size={out_osm.stat().st_size} bytes")
    return out_osm


def fallback_download_via_http(bbox_str: str, out_osm: Path) -> Path:
    """直接请求 Overpass HTTP 作为回退"""
    import urllib.request
    out_osm.parent.mkdir(parents=True, exist_ok=True)
    urls = [
        f"https://overpass-api.de/api/map?bbox={bbox_str}",
        f"https://overpass.kumi.systems/api/map?bbox={bbox_str}",
    ]
    last_err = None
    for u in urls:
        try:
            print(f"[INFO] 回退：HTTP 获取 {u}")
            with urllib.request.urlopen(u, timeout=180) as resp, open(out_osm, "wb") as f:
                shutil.copyfileobj(resp, f)
            if out_osm.exists() and out_osm.stat().st_size > 0:
                print(f"[OK] 已通过 HTTP 获取：{out_osm}  size={out_osm.stat().st_size} bytes")
                return out_osm
        except Exception as e:
            last_err = e
            print(f"[WARN] 获取失败：{e}")
    raise RuntimeError(f"Overpass HTTP 获取失败：{last_err}")


# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser("Fetch OSM and build SUMO net（兼容老版本 netconvert）")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--center", type=str, default="121.4737,31.2304",
                   help="中心点 lon,lat（默认：人民广场 121.4737,31.2304）")
    g.add_argument("--bbox", type=str,
                   help="直接给 bbox=minlon,minlat,maxlon,maxlat（经纬度）")
    ap.add_argument("--radius_km", type=float, default=5.0, help="半径（km），当使用 --center 时有效")
    ap.add_argument("--out_osm", type=str, default="net/shanghai_5km.osm.xml", help="导出的 OSM XML 路径")
    ap.add_argument("--out_net", type=str, default="net/shanghai_5km.net.xml", help="输出 SUMO 路网 .net.xml")
    ap.add_argument("--proj", type=str, default="auto",
                   help="投影：auto（默认自动选择）；或 'epsg:32651'；或 PROJ4 字符串（例如 '+proj=utm +zone=51 +datum=WGS84 +units=m +no_defs'）")
    ap.add_argument("--keep_vclass", type=str, default="passenger",
                   help="仅保留某类可行驶道路（默认 passenger；留空则不限制）")
    ap.add_argument("--remove_islands", action="store_true", default=True,
                   help="尝试移除孤立小子网（若 netconvert 支持）")
    ap.add_argument("--prune_small_nets", type=int, default=200,
                   help="尝试裁剪极小子网（米），若不支持将忽略")
    ap.add_argument("--use_typemap", action="store_true", default=True,
                   help="使用官方 OSM→SUMO 类型映射（默认开启）")
    ap.add_argument("--overwrite", action="store_true", help="若目标已存在则覆盖")
    ap.add_argument("--http_fallback", action="store_true", help="osmGet 失败时，尝试 HTTP 回退")
    args = ap.parse_args()

    osm_get, netconvert, sumo_home = check_sumo_env()
    help_txt = get_help_text(netconvert)

    # 解析 bbox
    if args.bbox:
        parts = [float(x.strip()) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("--bbox 需为 minlon,minlat,maxlon,maxlat")
        minlon, minlat, maxlon, maxlat = parts
    else:
        lon_str, lat_str = args.center.split(",")
        center_lon, center_lat = float(lon_str), float(lat_str)
        minlon, minlat, maxlon, maxlat = lonlat_bbox_from_center(center_lon, center_lat, args.radius_km)
    bbox_str = f"{minlon:.6f},{minlat:.6f},{maxlon:.6f},{maxlat:.6f}"
    print(f"[INFO] 使用 bbox: {bbox_str}  （center/r={args.center}/{args.radius_km}km）")

    # 输出路径
    out_osm = Path(args.out_osm); out_osm.parent.mkdir(parents=True, exist_ok=True)
    out_net = Path(args.out_net); out_net.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: 获取 OSM（优先 osmGet，必要时回退）
    if out_osm.exists() and not args.overwrite:
        print(f"[SKIP] 已存在 {out_osm}，跳过下载（如需重下加 --overwrite）")
    else:
        try:
            download_osm_with_osmget(osm_get, bbox_str, out_osm)
        except Exception as e:
            print(f"[WARN] osmGet 下载失败：{e}")
            if args.http_fallback:
                out_osm = fallback_download_via_http(bbox_str, out_osm)
            else:
                raise

    # Step 2: 生成路网（自动兼容参数）
    cmd = [netconvert,
           "--osm-files", str(out_osm),
           "--tls.guess", "true",
           "--roundabouts.guess", "true",
           "--ramps.guess", "true",
           "--junctions.join", "true",
           "--output-file", str(out_net)]

    # 投影
    if args.proj == "auto":
        if supports(help_txt, "--proj.epsg"):
            cmd += ["--proj.epsg", "32651"]  # UTM 51N
        else:
            # 老版本通吃：PROJ4 字符串
            cmd += ["--proj", "+proj=utm +zone=51 +datum=WGS84 +units=m +no_defs"]
    else:
        if args.proj.lower().startswith("epsg:") and supports(help_txt, "--proj.epsg"):
            cmd += ["--proj.epsg", args.proj.split(":", 1)[1]]
        else:
            cmd += ["--proj", args.proj]

    # 仅保留某类道路
    if args.keep_vclass:
        cmd += ["--keep-edges.by-vclass", args.keep_vclass]

    # 类型映射
    if args.use_typemap:
        typemap = sumo_home / "data" / "typemap" / "osmNetconvert.typ.xml"
        if typemap.exists():
            cmd += ["--type-files", str(typemap)]
        else:
            print(f"[WARN] 未找到 typemap：{typemap}，跳过。")

    # 清理选项（仅在支持时添加）
    if args.remove_islands and supports(help_txt, "--remove-edges.islands"):
        cmd += ["--remove-edges.islands"]
    elif args.remove_islands:
        print("[WARN] netconvert 不支持 --remove-edges.islands，已跳过。")

    if args.prune_small_nets and args.prune_small_nets > 0 and supports(help_txt, "--prune.small-nets"):
        cmd += ["--prune.small-nets", str(args.prune_small_nets)]
    elif args.prune_small_nets and args.prune_small_nets > 0:
        print("[WARN] netconvert 不支持 --prune.small-nets，已跳过。")

    # 执行 netconvert
    if out_net.exists() and not args.overwrite:
        print(f"[SKIP] 已存在 {out_net}，若需重建加 --overwrite")
    else:
        run(cmd, check=True)

    # Step 3: 自检（米坐标 BBox）
    try:
        from sumolib.net import readNet
        net = readNet(str(out_net))
        (minx, miny), (maxx, maxy) = net.getBBoxXY()
        print(f"[OK] 生成路网：{out_net}")
        print(f"[OK] Net BBox (meters): (({minx:.1f},{miny:.1f}),({maxx:.1f},{maxy:.1f}))  "
              f"width={maxx-minx:.1f}m  height={maxy-miny:.1f}m")
        if max(maxx - minx, maxy - miny) < 1000:
            print("[WARN] 尺度偏小，可能未正确投影为米（请检查 --proj 参数）。")
    except Exception as e:
        print(f"[WARN] 无法读取 net bbox：{e}")
        print("       可用 sumo-gui/netedit 手动打开检查。")


if __name__ == "__main__":
    main()
