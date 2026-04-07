#!/usr/bin/env python3
"""
将 lidar 渲染输出的 .npy 点云转换为 .ply。

默认假设 npy 为 float 数组，形状为:
- [N, 4] -> (x, y, z, intensity)
也支持:
- [N, 3] -> (x, y, z)
- 结构化数组（带字段名 x/y/z/intensity 或 r/g/b）
"""

from __future__ import annotations

import argparse
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np


def _infer_columns(
    arr: np.ndarray,
) -> Tuple[Tuple[int, int, int], Optional[int], Optional[Tuple[int, int, int]]]:
    """根据 ndarray 的形状推断 xyz / intensity / rgb 的列索引。"""
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"期望输入为二维点表 [N, D] 且 D>=3，实际为 {arr.shape}")
    xyz = (0, 1, 2)
    intensity = 3 if arr.shape[1] >= 4 else None
    rgb = (3, 4, 5) if arr.shape[1] >= 6 else None
    return xyz, intensity, rgb


def _load_npy(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """加载 npy，兼容普通 ndarray 与结构化数组，返回 (xyz/intensity/rgb) 字典。"""
    arr = np.load(path, allow_pickle=False)

    out: Dict[str, np.ndarray] = {}

    if isinstance(arr, np.ndarray) and arr.dtype.names:
        names = set(arr.dtype.names)
        required_xyz = {"x", "y", "z"}
        if not required_xyz.issubset(names):
            raise ValueError(
                f"结构化npy缺少字段 {sorted(required_xyz - names)}，现有字段={sorted(names)}"
            )
        out["xyz"] = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(
            np.float32, copy=False
        )

        if "intensity" in names:
            out["intensity"] = np.asarray(arr["intensity"], dtype=np.float32)

        if {"r", "g", "b"}.issubset(names):
            rgb = np.stack([arr["r"], arr["g"], arr["b"]], axis=-1)
            out["rgb"] = np.asarray(rgb)
        return arr, out

    if not (isinstance(arr, np.ndarray) and arr.ndim == 2):
        raise ValueError(
            f"不支持的npy内容类型/形状: type={type(arr)} shape={getattr(arr, 'shape', None)}"
        )

    xyz_cols, intensity_col, rgb_cols = _infer_columns(arr)
    xyz = arr[:, list(xyz_cols)].astype(np.float32, copy=False)
    out["xyz"] = xyz

    if intensity_col is not None:
        out["intensity"] = arr[:, intensity_col].astype(np.float32, copy=False)

    if rgb_cols is not None:
        out["rgb"] = arr[:, list(rgb_cols)]

    return arr, out


def _intensity_to_rgb(intensity: np.ndarray, clip: Tuple[float, float]) -> np.ndarray:
    """把强度映射为灰度 RGB（0-255），便于在点云工具里直观查看。"""
    lo, hi = float(clip[0]), float(clip[1])
    x = intensity.astype(np.float32, copy=False)
    x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    g = (x * 255.0 + 0.5).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def write_ply_ascii(
    out_path: str,
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    rgb_u8: Optional[np.ndarray] = None,
) -> None:
    """以 ASCII PLY 写入点云。"""
    n = int(xyz.shape[0])
    lines: List[str] = []
    lines.append("ply")
    lines.append("format ascii 1.0")
    lines.append(f"element vertex {n}")
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    if intensity is not None:
        lines.append("property float intensity")
    if rgb_u8 is not None:
        lines.append("property uchar red")
        lines.append("property uchar green")
        lines.append("property uchar blue")
    lines.append("end_header")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        if intensity is None and rgb_u8 is None:
            np.savetxt(f, xyz, fmt="%.6f %.6f %.6f")
            return

        for i in range(n):
            row = [f"{xyz[i, 0]:.6f}", f"{xyz[i, 1]:.6f}", f"{xyz[i, 2]:.6f}"]
            if intensity is not None:
                row.append(f"{float(intensity[i]):.6f}")
            if rgb_u8 is not None:
                r, g, b = rgb_u8[i].tolist()
                row.extend([str(int(r)), str(int(g)), str(int(b))])
            f.write(" ".join(row) + "\n")


def write_ply_binary_little_endian(
    out_path: str,
    xyz: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    rgb_u8: Optional[np.ndarray] = None,
) -> None:
    """以 binary_little_endian PLY 写入点云。"""
    n = int(xyz.shape[0])
    header: List[str] = []
    header.append("ply")
    header.append("format binary_little_endian 1.0")
    header.append(f"element vertex {n}")
    header.append("property float x")
    header.append("property float y")
    header.append("property float z")
    if intensity is not None:
        header.append("property float intensity")
    if rgb_u8 is not None:
        header.append("property uchar red")
        header.append("property uchar green")
        header.append("property uchar blue")
    header.append("end_header")
    header_bytes = ("\n".join(header) + "\n").encode("ascii")

    xyz_f32 = np.asarray(xyz, dtype=np.float32)
    intensity_f32 = (
        None if intensity is None else np.asarray(intensity, dtype=np.float32)
    )
    rgb = None if rgb_u8 is None else np.asarray(rgb_u8, dtype=np.uint8)

    fmt = "<fff"
    if intensity_f32 is not None:
        fmt += "f"
    if rgb is not None:
        fmt += "BBB"
    pack = struct.Struct(fmt).pack

    with open(out_path, "wb") as f:
        f.write(header_bytes)
        for i in range(n):
            x, y, z = xyz_f32[i].tolist()
            if intensity_f32 is None and rgb is None:
                f.write(pack(x, y, z))
            elif intensity_f32 is not None and rgb is None:
                f.write(pack(x, y, z, float(intensity_f32[i])))
            elif intensity_f32 is None and rgb is not None:
                r, g, b = rgb[i].tolist()
                f.write(pack(x, y, z, int(r), int(g), int(b)))
            else:
                r, g, b = rgb[i].tolist()
                f.write(pack(x, y, z, float(intensity_f32[i]), int(r), int(g), int(b)))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="Convert lidar npy point cloud to ply")
    p.add_argument("--input", "-i", type=str, required=True, help="输入 .npy 文件路径")
    p.add_argument(
        "--output", "-o", type=str, default="", help="输出 .ply 文件路径（默认同名）"
    )
    p.add_argument(
        "--binary",
        action="store_true",
        help="输出 binary_little_endian PLY（默认 ASCII）",
    )
    p.add_argument(
        "--rgb-from-intensity",
        action="store_true",
        help="把 intensity 映射为灰度 RGB 写入 ply，便于可视化（不改变 xyz/intensity）",
    )
    p.add_argument(
        "--intensity-clip",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("MIN", "MAX"),
        help="rgb-from-intensity 的强度归一化范围",
    )
    return p.parse_args()


def main() -> None:
    """脚本入口：读取 npy，写出 ply。"""
    args = parse_args()
    in_path = args.input
    if not os.path.isfile(in_path):
        raise FileNotFoundError(in_path)

    out_path = args.output
    if not out_path:
        base, _ = os.path.splitext(in_path)
        out_path = base + ".ply"

    _raw, data = _load_npy(in_path)
    xyz = data["xyz"]
    intensity = data.get("intensity", None)

    rgb_u8 = None
    if "rgb" in data:
        rgb = np.asarray(data["rgb"])
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"rgb 期望 [N,3]，实际 {rgb.shape}")
        rgb_u8 = rgb

    if args.rgb_from_intensity and intensity is not None:
        rgb_u8 = _intensity_to_rgb(intensity, clip=tuple(args.intensity_clip))

    if args.binary:
        write_ply_binary_little_endian(
            out_path, xyz=xyz, intensity=intensity, rgb_u8=rgb_u8
        )
    else:
        write_ply_ascii(out_path, xyz=xyz, intensity=intensity, rgb_u8=rgb_u8)

    print(
        f"已写出: {out_path}  (N={xyz.shape[0]}, intensity={'yes' if intensity is not None else 'no'}, rgb={'yes' if rgb_u8 is not None else 'no'})"
    )


if __name__ == "__main__":
    main()
