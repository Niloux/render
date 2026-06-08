"""Vehicle model library loading."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from models import GaussianComponent


VEHICLE_LIBRARY_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPONENT_REQUIRED_KEYS = (
    "xyz",
    "feature_dc",
    "feature_rest",
    "scaling",
    "rotation",
    "opacity",
)


@dataclass
class VehicleModelLibrary:
    path: Path
    camera_actors: List[GaussianComponent]
    lidar_actors: List[GaussianComponent]

    @classmethod
    def load_from_directory(
        cls,
        library_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "VehicleModelLibrary":
        path = Path(library_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"车模库目录不存在: {path}")

        manifest = _load_manifest(path / MANIFEST_NAME)
        vehicles = _manifest_vehicles(manifest)

        camera_actors: List[GaussianComponent] = []
        lidar_actors: List[GaussianComponent] = []
        for vehicle_id, manifest_entry in vehicles.items():
            if not _is_enabled(manifest_entry):
                continue
            vehicle_data = _load_vehicle_file(path, vehicle_id, manifest_entry)

            camera_actors.append(
                _build_vehicle_component(vehicle_id, vehicle_data, "camera", device)
            )
            lidar_actors.append(
                _build_vehicle_component(vehicle_id, vehicle_data, "lidar", device)
            )

        if not camera_actors:
            raise ValueError(f"车模库{path}没有启用的车辆模型")

        return cls(path=path, camera_actors=camera_actors, lidar_actors=lidar_actors)

    @property
    def camera_actor_map(self) -> Dict[str, GaussianComponent]:
        return {actor.name: actor for actor in self.camera_actors}

    @property
    def lidar_actor_map(self) -> Dict[str, GaussianComponent]:
        return {actor.name: actor for actor in self.lidar_actors}


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"车模库缺少{MANIFEST_NAME}: {path}")
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"车模库manifest必须是dict，实际为{type(manifest)!r}")
    version = int(manifest.get("version", 0))
    if version != VEHICLE_LIBRARY_VERSION:
        raise ValueError(
            f"不支持的车模库版本: {version}，期望为{VEHICLE_LIBRARY_VERSION}"
        )
    return manifest


def _manifest_vehicles(manifest: dict) -> Dict[str, dict]:
    vehicles = manifest.get("vehicles")
    if not isinstance(vehicles, dict) or not vehicles:
        raise ValueError("车模库manifest必须包含非空vehicles字典")

    out: Dict[str, dict] = {}
    for vehicle_id, entry in vehicles.items():
        if not isinstance(vehicle_id, str) or not vehicle_id:
            raise ValueError(f"车模id必须是非空字符串，实际为{vehicle_id!r}")
        if not isinstance(entry, dict):
            raise ValueError(f"车模{vehicle_id}的manifest条目必须是dict")
        out[vehicle_id] = entry
    return out


def _is_enabled(manifest_entry: dict) -> bool:
    return bool(manifest_entry.get("enabled", True))


def _load_vehicle_file(
    library_path: Path,
    vehicle_id: str,
    manifest_entry: dict,
) -> dict:
    file_name = manifest_entry.get("file")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(f"车模{vehicle_id}的manifest条目缺少file字段")
    vehicle_path = library_path / file_name
    checkpoint = torch.load(vehicle_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"车模{vehicle_id}文件必须是dict，实际为{type(checkpoint)!r}")
    version = int(checkpoint.get("version", 0))
    if version != VEHICLE_LIBRARY_VERSION:
        raise ValueError(
            f"车模{vehicle_id}文件版本为{version}，期望为{VEHICLE_LIBRARY_VERSION}"
        )
    stored_id = checkpoint.get("id")
    if stored_id != vehicle_id:
        raise ValueError(f"车模文件id不匹配: manifest={vehicle_id} file={stored_id}")
    return checkpoint


def _build_vehicle_component(
    vehicle_id: str,
    vehicle_data: Dict[str, Any],
    group: str,
    device: Optional[torch.device],
) -> GaussianComponent:
    component_data = vehicle_data.get(group)
    if not isinstance(component_data, dict):
        raise ValueError(f"车模{vehicle_id}缺少{group}点云")
    missing = [key for key in COMPONENT_REQUIRED_KEYS if key not in component_data]
    if missing:
        raise ValueError(f"车模{vehicle_id}的{group}点云缺少字段: {missing}")

    component = GaussianComponent(
        name=vehicle_id,
        xyz=component_data["xyz"],
        feature_dc=component_data["feature_dc"],
        feature_rest=component_data["feature_rest"],
        scaling=component_data["scaling"],
        rotation=component_data["rotation"],
        opacity=component_data["opacity"],
        semantic=component_data.get("semantic"),
    )
    if not component.validate():
        raise ValueError(f"车模{vehicle_id}的{group}点云数据验证失败")
    if device is not None:
        component = component.to_device(device)
    return component
