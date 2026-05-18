#!/usr/bin/env python3
"""Inspect and validate render checkpoint files."""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import GSModel  # noqa: E402


def format_shape(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"{tuple(value.shape)} {value.dtype}"
    return type(value).__name__


def format_count(value: int) -> str:
    return f"{value:,}"


def print_raw_structure(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    print("\nRaw Checkpoint")
    print(f"- type: {type(checkpoint).__name__}")
    if not isinstance(checkpoint, dict):
        return

    print(f"- keys: {len(checkpoint)}")
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            print(f"- {key}: Tensor {format_shape(value)}")
        elif isinstance(value, dict):
            print(f"- {key}: Dict ({len(value)} keys)")
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {format_shape(sub_value)}")
        else:
            print(f"- {key}: {type(value).__name__}")


def print_model_summary(model: GSModel, path: Path, show_components: bool) -> None:
    background = model.get_component("background")
    sky = model.get_component("sky")
    actors = model.get_components_by_type("obj")

    print("Checkpoint")
    print(f"- path: {path}")
    print(f"- group: {model.group}")
    print(f"- iteration: {model.iteration}")
    print(f"- components: {len(model.components)}")
    print(f"- total_points: {format_count(model.total_points)}")
    print(f"- estimated_tensor_memory: {model.total_memory:.2f} MB")

    print("\nRender Readiness")
    print(f"- background: {'yes' if background is not None else 'no'}")
    print(f"- map_center: {format_shape(model.map_center)}")
    print(f"- sky_component: {'yes' if sky is not None else 'no'}")
    print(f"- sky_cubemap: {format_shape(model.sky_cubemap) if model.sky_cubemap is not None else 'no'}")
    print(f"- actor_components: {len(actors)}")
    print(f"- rgb_decoder: {'yes' if model.rgb_decoder_state is not None else 'no'}")
    print(f"- lidar_raster_pts: {format_shape(model.raster_pts) if model.raster_pts is not None else 'no'}")
    print(f"- mlp_decoder: {'yes' if model.mlp_decoder_state is not None else 'no'}")

    if not show_components:
        return

    print("\nComponents")
    for name in sorted(model.component_names):
        component = model.get_component(name)
        if component is None:
            continue
        color_channels = int(component.feature_dc.shape[-1])
        print(
            f"- {name}: points={format_count(component.num_points)}, "
            f"channels={color_channels}, memory={component.memory_usage:.2f} MB"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a render .pth checkpoint against current GSModel loading rules."
    )
    parser.add_argument("path", type=Path, help="Path to the .pth checkpoint")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also print the raw top-level checkpoint structure.",
    )
    parser.add_argument(
        "--components",
        action="store_true",
        help="Print one summary line per Gaussian component.",
    )
    parser.add_argument(
        "--group",
        choices=("camera", "lidar", "all"),
        default="all",
        help="Which new-format checkpoint group to validate.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip invalid components instead of failing on component validation errors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.path.expanduser().resolve()

    try:
        groups = ("camera", "lidar") if args.group == "all" else (args.group,)
        for index, group in enumerate(groups):
            if index:
                print()
            model = GSModel.load_from_pth(
                path, group=group, strict=not args.non_strict
            )
            print_model_summary(model, path, show_components=args.components)
        if args.raw:
            print_raw_structure(path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
