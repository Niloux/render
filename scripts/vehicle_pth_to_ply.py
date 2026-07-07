#!/usr/bin/env python3
"""Convert one vehicle-library pth file to PLY."""

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import torch


SH_C0 = 0.28209479177387814
REQUIRED_KEYS = (
    "xyz",
    "feature_dc",
    "feature_rest",
    "scaling",
    "rotation",
    "opacity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a vehicle-library pth file to an ASCII PLY file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Vehicle pth file, for example vehicles/car_0001.pth.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PLY path. Defaults to <input>_<group>.ply.",
    )
    parser.add_argument(
        "--group",
        choices=("camera", "lidar"),
        default="camera",
        help="Vehicle branch to export. Default: camera.",
    )
    parser.add_argument(
        "--format",
        choices=("gaussian", "pointcloud"),
        default="gaussian",
        help=(
            "gaussian preserves 3DGS attributes; pointcloud writes xyz/rgb only. "
            "Default: gaussian."
        ),
    )
    return parser.parse_args()


def as_cpu_float(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}")
    return tensor.detach().cpu().float().contiguous()


def load_vehicle_component(path: Path, group: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"vehicle pth must be a dict, got {type(checkpoint)!r}")

    component = checkpoint.get(group)
    if not isinstance(component, dict):
        raise KeyError(f"vehicle pth is missing the {group!r} branch")

    missing = [key for key in REQUIRED_KEYS if key not in component]
    if missing:
        raise KeyError(f"{group} branch is missing required fields: {missing}")

    out = {key: as_cpu_float(component[key], key) for key in REQUIRED_KEYS}
    semantic = component.get("semantic")
    if isinstance(semantic, torch.Tensor) and semantic.numel() > 0:
        out["semantic"] = as_cpu_float(semantic, "semantic")
    return out


def validate_component(component: dict) -> int:
    xyz = component["xyz"]
    feature_dc = component["feature_dc"]
    feature_rest = component["feature_rest"]
    scaling = component["scaling"]
    rotation = component["rotation"]
    opacity = component["opacity"]

    n = xyz.shape[0]
    expected = {
        "xyz": (n, 3),
        "feature_dc": (n, 1, feature_dc.shape[2] if feature_dc.ndim == 3 else -1),
        "scaling": (n, 3),
        "rotation": (n, 4),
        "opacity": (n, 1),
    }
    for key, shape in expected.items():
        if tuple(component[key].shape) != shape:
            raise ValueError(f"{key} shape must be {shape}, got {tuple(component[key].shape)}")
    if feature_dc.shape[2] < 3:
        raise ValueError(f"feature_dc must have at least 3 channels, got {feature_dc.shape[2]}")
    if feature_rest.ndim != 3 or feature_rest.shape[0] != n:
        raise ValueError(
            "feature_rest shape must be [N, K, C], "
            f"got {tuple(feature_rest.shape)}"
        )
    if feature_rest.shape[2] != feature_dc.shape[2]:
        raise ValueError(
            "feature_rest channel count must match feature_dc, "
            f"got {feature_rest.shape[2]} vs {feature_dc.shape[2]}"
        )
    semantic = component.get("semantic")
    if semantic is not None and (semantic.ndim != 2 or semantic.shape[0] != n):
        raise ValueError(f"semantic shape must be [N, K], got {tuple(semantic.shape)}")
    return n


def default_output_path(input_path: Path, group: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{group}.ply")


def dc_to_rgb(feature_dc: torch.Tensor) -> torch.Tensor:
    rgb = (feature_dc[:, 0, :3] * SH_C0 + 0.5).clamp(0.0, 1.0)
    return (rgb * 255.0).round().to(torch.uint8)


def gaussian_properties(component: dict) -> List[Tuple[str, torch.Tensor]]:
    n = component["xyz"].shape[0]
    feature_dc = component["feature_dc"][:, 0, :]
    feature_rest = component["feature_rest"].permute(0, 2, 1).reshape(n, -1)

    properties: List[Tuple[str, torch.Tensor]] = [
        ("x", component["xyz"][:, 0]),
        ("y", component["xyz"][:, 1]),
        ("z", component["xyz"][:, 2]),
        ("nx", torch.zeros(n)),
        ("ny", torch.zeros(n)),
        ("nz", torch.zeros(n)),
    ]
    properties.extend((f"f_dc_{idx}", feature_dc[:, idx]) for idx in range(feature_dc.shape[1]))
    properties.extend(
        (f"f_rest_{idx}", feature_rest[:, idx]) for idx in range(feature_rest.shape[1])
    )
    properties.append(("opacity", component["opacity"][:, 0]))
    properties.extend(
        (f"scale_{idx}", component["scaling"][:, idx])
        for idx in range(component["scaling"].shape[1])
    )
    properties.extend(
        (f"rot_{idx}", component["rotation"][:, idx])
        for idx in range(component["rotation"].shape[1])
    )
    semantic = component.get("semantic")
    if semantic is not None:
        properties.extend((f"semantic_{idx}", semantic[:, idx]) for idx in range(semantic.shape[1]))
    return properties


def pointcloud_properties(component: dict) -> List[Tuple[str, torch.Tensor]]:
    rgb = dc_to_rgb(component["feature_dc"])
    return [
        ("x", component["xyz"][:, 0]),
        ("y", component["xyz"][:, 1]),
        ("z", component["xyz"][:, 2]),
        ("red", rgb[:, 0]),
        ("green", rgb[:, 1]),
        ("blue", rgb[:, 2]),
    ]


def ply_property_lines(properties: Iterable[Tuple[str, torch.Tensor]]) -> List[str]:
    lines = []
    for name, values in properties:
        dtype = "uchar" if values.dtype == torch.uint8 else "float"
        lines.append(f"property {dtype} {name}")
    return lines


def write_ascii_ply(path: Path, n: int, properties: List[Tuple[str, torch.Tensor]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [values.tolist() for _, values in properties]
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        for line in ply_property_lines(properties):
            f.write(f"{line}\n")
        f.write("end_header\n")

        for row in zip(*columns):
            f.write(" ".join(str(value) for value in row))
            f.write("\n")


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path, args.group)
    )

    component = load_vehicle_component(input_path, args.group)
    n = validate_component(component)
    properties = (
        gaussian_properties(component)
        if args.format == "gaussian"
        else pointcloud_properties(component)
    )
    write_ascii_ply(output_path, n, properties)
    print(
        f"Wrote {output_path} from {input_path} "
        f"group={args.group} format={args.format} points={n}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
