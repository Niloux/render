#!/usr/bin/env python3
"""Extract one paired camera/lidar actor from a scene checkpoint into a vehicle library."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import GaussianComponent, GSModel  # noqa: E402
from vehicle_library import MANIFEST_NAME, VEHICLE_LIBRARY_VERSION  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a paired camera/lidar obj_* component into a vehicle library."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source scene checkpoint, for example 0605.pth.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Vehicle library directory. The script writes one vehicle pth and manifest.json.",
    )
    parser.add_argument(
        "--actor",
        default="obj_009",
        help="Source actor component name to extract.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Vehicle id in the library. Defaults to --actor.",
    )
    return parser.parse_args()


def component_to_data(component: GaussianComponent) -> dict:
    data = {
        "xyz": component.xyz.detach().cpu(),
        "feature_dc": component.feature_dc.detach().cpu(),
        "feature_rest": component.feature_rest.detach().cpu(),
        "scaling": component.scaling.detach().cpu(),
        "rotation": component.rotation.detach().cpu(),
        "opacity": component.opacity.detach().cpu(),
    }
    if component.semantic is not None:
        data["semantic"] = component.semantic.detach().cpu()
    return data


def required_component(model: GSModel, name: str, group: str) -> GaussianComponent:
    component = model.get_component(name)
    if component is None:
        raise KeyError(f"{model.group or group}子模型中不存在组件: {name}")
    return component


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"version": VEHICLE_LIBRARY_VERSION, "vehicles": {}}
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest必须是dict，实际为{type(manifest)!r}")
    if int(manifest.get("version", 0)) != VEHICLE_LIBRARY_VERSION:
        raise ValueError(
            f"不支持的manifest版本: {manifest.get('version')}，"
            f"期望为{VEHICLE_LIBRARY_VERSION}"
        )
    vehicles = manifest.setdefault("vehicles", {})
    if not isinstance(vehicles, dict):
        raise ValueError("manifest中的vehicles必须是dict")
    return manifest


def write_manifest(path: Path, manifest: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    vehicle_id = args.name or args.actor
    vehicle_file = f"{vehicle_id}.pth"

    camera_model = GSModel.load_from_pth(source, group="camera")
    lidar_model = GSModel.load_from_pth(source, group="lidar")
    camera_component = required_component(camera_model, args.actor, "camera")
    lidar_component = required_component(lidar_model, args.actor, "lidar")

    checkpoint = {
        "version": VEHICLE_LIBRARY_VERSION,
        "id": vehicle_id,
        "camera": component_to_data(camera_component),
        "lidar": component_to_data(lidar_component),
        "meta": {
            "source": str(source),
            "source_actor": args.actor,
            "camera_points": camera_component.num_points,
            "lidar_points": lidar_component.num_points,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / vehicle_file
    torch.save(checkpoint, output)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    manifest["vehicles"][vehicle_id] = {
        "file": vehicle_file,
        "enabled": True,
        "source_model": str(source),
        "source_actor": args.actor,
        "camera_points": camera_component.num_points,
        "lidar_points": lidar_component.num_points,
    }
    write_manifest(manifest_path, manifest)
    print(
        f"Wrote {output} and {manifest_path}: {vehicle_id} "
        f"camera_points={camera_component.num_points:,} "
        f"lidar_points={lidar_component.num_points:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
