"""Benchmark scenario, sensor parsing, and output helpers."""

import json
import os
import statistics
import time
from typing import TYPE_CHECKING, Any, List, Tuple, TypedDict

import numpy as np
import torch

from data_types import Camera, FrameParams, InitParams, Lidar, Vehicle
from util import save_colors_as_png

if TYPE_CHECKING:
    from render_manager import RenderManager


class RenderConfig(TypedDict):
    render_camera: bool
    render_lidar: bool


class BenchmarkConfig(TypedDict):
    model_path: str
    warmup_frames: int
    benchmark_frames: int
    save_first_frame: bool
    output_dir: str
    sensors_json: str
    camera_count: int
    resolution: Tuple[int, int]
    lidar_count: int


class BenchmarkStats(TypedDict):
    total_frames: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    fps: float


STANDARD_EXTRINSICS: List[List[float]] = [
    [
        -4.588266398430291063e-03,
        -3.413667297520365119e-03,
        9.999836472098125872e-01,
        1.544154267170511075e00,
    ],
    [
        -9.999632354307952387e-01,
        -7.228375769820964344e-03,
        -4.612848415714690224e-03,
        -2.315740942895095494e-02,
    ],
    [
        7.244004295493749329e-03,
        -9.999680482191979358e-01,
        -3.380376081852865672e-03,
        2.115612062706179408e00,
    ],
    [0.0, 0.0, 0.0, 1.0],
]

STANDARD_INTRINSICS: List[List[float]] = [
    [2.084604312956008926e03, 0.0, 9.334067577078354816e02],
    [0.0, 2.084604312956008926e03, 6.650223418347507049e02],
    [0.0, 0.0, 1.0],
]

LIDAR_EXTRINSICS: List[List[float]] = [
    [-0.852646995188958, -0.522486863163586, 0.000761194269898, 1.430000000000000],
    [0.522487398378445, -0.852645663278401, 0.001513746431570, 0.000000000000000],
    [-0.000141883631515, 0.001688405760096, 0.999998564576482, 2.184000000000000],
    [0, 0, 0, 1],
]


def validate_matrix(mat: Any, rows: int, cols: int, name: str) -> List[List[float]]:
    if not isinstance(mat, list) or len(mat) != rows:
        raise ValueError(f"{name}期望为{rows}x{cols}二维列表")
    out: List[List[float]] = []
    for r in range(rows):
        row = mat[r]
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(f"{name}第{r}行期望长度为{cols}")
        out.append([float(x) for x in row])
    return out


def save_point_cloud_as_ply(
    save_path: str, point_cloud: torch.Tensor | np.ndarray
) -> None:
    if isinstance(point_cloud, torch.Tensor):
        points = point_cloud.detach().cpu().numpy()
    else:
        points = np.asarray(point_cloud)

    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError(
            f"point_cloud 期望形状为 [N,3] 或 [N,4]，实际为 {points.shape}"
        )

    points = points.astype(np.float32, copy=False)
    points = points[np.isfinite(points).all(axis=1)]
    has_intensity = points.shape[1] == 4
    if has_intensity:
        dtype = np.dtype([
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
        ])
        properties = "property float x\nproperty float y\nproperty float z\nproperty float intensity\n"
    else:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        properties = "property float x\nproperty float y\nproperty float z\n"

    packed = np.empty(points.shape[0], dtype=dtype)
    packed["x"] = points[:, 0]
    packed["y"] = points[:, 1]
    packed["z"] = points[:, 2]
    if has_intensity:
        packed["intensity"] = points[:, 3]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {packed.shape[0]}\n"
        f"{properties}"
        "end_header\n"
    ).encode("ascii")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(header)
        f.write(packed.tobytes())


def load_cameras_from_specs(
    specs: Any, default_width: int, default_height: int
) -> List[Camera]:
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("cameras 内容必须是列表")

    cameras: List[Camera] = []
    for idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"camera spec 必须是对象(dict)，实际为{type(spec)}")
        cam_id = str(spec.get("id", f"camera{idx + 1}"))
        cameras.append(
            Camera(
                cam_id,
                validate_matrix(
                    spec.get("extrinsics", STANDARD_EXTRINSICS),
                    4,
                    4,
                    f"{cam_id}.extrinsics",
                ),
                validate_matrix(
                    spec.get("intrinsics", STANDARD_INTRINSICS),
                    3,
                    3,
                    f"{cam_id}.intrinsics",
                ),
                int(spec.get("width", default_width)),
                int(spec.get("height", default_height)),
            )
        )
    return cameras


def load_lidars_from_specs(specs: Any) -> List[Lidar]:
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("lidars 内容必须是列表")

    lidars: List[Lidar] = []
    for idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"lidar spec 必须是对象(dict)，实际为{type(spec)}")
        lidar_id = str(spec.get("id", f"lidar{idx + 1}"))
        lidars.append(
            Lidar(
                lidar_id,
                validate_matrix(
                    spec.get("extrinsics", LIDAR_EXTRINSICS),
                    4,
                    4,
                    f"{lidar_id}.extrinsics",
                ),
                azimuth_resolution=float(spec.get("azimuth_resolution", 0.140625)),
                min_azimuth=float(spec.get("min_azimuth", -180.0)),
                max_azimuth=float(spec.get("max_azimuth", 180.0)),
                n_elevation_channels=int(spec.get("n_elevation_channels", 64)),
                min_elevation=float(spec.get("min_elevation", -17.55)),
                max_elevation=float(spec.get("max_elevation", 2.5)),
                near_plane=float(spec.get("near_plane", 0.01)),
                far_plane=float(spec.get("far_plane", 1e10)),
                tile_width=int(spec.get("tile_width", 32)),
                tile_height=int(spec.get("tile_height", 8)),
            )
        )
    return lidars


def load_sensors_from_json(
    path: str, default_width: int, default_height: int
) -> Tuple[List[Camera], List[Lidar]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("sensors_json 根节点必须是对象(dict)")

    cameras = load_cameras_from_specs(
        data.get("cameras"), default_width=default_width, default_height=default_height
    )
    lidars = load_lidars_from_specs(data.get("lidars"))
    if not cameras and not lidars:
        raise ValueError("sensors_json 必须至少包含 cameras 或 lidars 之一")
    return cameras, lidars


def create_test_cameras(count: int, width: int, height: int) -> List[Camera]:
    return [
        Camera(
            f"camera{i + 1}", STANDARD_EXTRINSICS, STANDARD_INTRINSICS, width, height
        )
        for i in range(count)
    ]


def create_test_lidars(count: int) -> List[Lidar]:
    return [
        Lidar(
            f"lidar{i + 1}",
            LIDAR_EXTRINSICS,
            azimuth_resolution=0.13392857142857142,
            min_azimuth=-180.0,
            max_azimuth=180.0,
            n_elevation_channels=64,
            min_elevation=-18.557777404785156,
            max_elevation=3.3896429538726807,
            tile_width=64,
            tile_height=4,
        )
        for i in range(count)
    ]


def create_test_scenario(
    config: BenchmarkConfig, render_config: RenderConfig
) -> Tuple[InitParams, FrameParams]:
    cameras: List[Camera] = []
    lidars: List[Lidar] = []

    if config.get("sensors_json") and (
        render_config["render_camera"] or render_config["render_lidar"]
    ):
        loaded_cameras, loaded_lidars = load_sensors_from_json(
            config["sensors_json"],
            default_width=config["resolution"][0],
            default_height=config["resolution"][1],
        )
        if render_config["render_camera"]:
            cameras = loaded_cameras
            if not cameras:
                raise ValueError("启用了相机渲染，但 sensors_json 中未提供 cameras")
        if render_config["render_lidar"]:
            lidars = loaded_lidars
            if not lidars:
                raise ValueError("启用了激光雷达渲染，但 sensors_json 中未提供 lidars")
    else:
        if render_config["render_camera"]:
            cameras = create_test_cameras(
                config["camera_count"], config["resolution"][0], config["resolution"][1]
            )
        if render_config["render_lidar"]:
            lidars = create_test_lidars(config["lidar_count"])

    init_params = InitParams(
        cameras=cameras,
        lidars=lidars,
        render_camera=render_config["render_camera"],
        render_lidar=render_config["render_lidar"],
    )
    frame_params = FrameParams(
        ego_trajectory=[8073.322207, 4679.904873, 48.617000],
        ego_yaw=0.02501,
        env_vehicles=[
            Vehicle([8103.21911375, 4680.84724959, 49.77610101], 0, "obj_015")
        ],
        timestamp=20260401,
    )
    return init_params, frame_params


def run_benchmark(
    render_manager: "RenderManager",
    frame_params: FrameParams,
    config: BenchmarkConfig,
    render_config: RenderConfig,
) -> BenchmarkStats:
    print("开始预热阶段...")
    for i in range(config["warmup_frames"]):
        render_manager.render_frame(frame_params)
        if i % 5 == 0:
            print(f"预热进度: {i + 1}/{config['warmup_frames']}")

    print("开始性能测试...")
    render_times: List[float] = []
    if config["save_first_frame"]:
        os.makedirs(config["output_dir"], exist_ok=True)

    for i in range(config["benchmark_frames"]):
        t0 = time.perf_counter()
        frame_resp = render_manager.render_frame(frame_params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        render_time = time.perf_counter() - t0
        render_times.append(render_time)

        if i == 0 and config["save_first_frame"]:
            save_first_frame(frame_resp, config, render_config)
        if (i + 1) % 20 == 0:
            print(
                f"测试进度: {i + 1}/{config['benchmark_frames']} - "
                f"当前帧耗时: {render_time * 1000:.2f}ms"
            )

    return {
        "total_frames": len(render_times),
        "mean_ms": statistics.mean(render_times) * 1000,
        "median_ms": statistics.median(render_times) * 1000,
        "min_ms": min(render_times) * 1000,
        "max_ms": max(render_times) * 1000,
        "std_ms": statistics.stdev(render_times) * 1000
        if len(render_times) > 1
        else 0.0,
        "fps": 1.0 / statistics.mean(render_times),
    }


def save_first_frame(
    frame_resp, config: BenchmarkConfig, render_config: RenderConfig
) -> None:
    if render_config["render_camera"] and frame_resp.images:
        print(f"正在保存相机图像到 {config['output_dir']}...")
        try:
            save_colors_as_png(frame_resp.images, output_dir=config["output_dir"])
        except Exception as exc:
            print(f"保存相机图像失败: {exc}")

    if render_config["render_lidar"] and frame_resp.lidars:
        print(f"正在保存激光雷达点云到 {config['output_dir']}...")
        for lidar_id, lidar_data in frame_resp.lidars.items():
            save_path = os.path.join(config["output_dir"], f"{lidar_id}.ply")
            save_point_cloud_as_ply(save_path, lidar_data)

    print("第一帧结果处理完成")


def print_benchmark_results(stats: BenchmarkStats, render_config: RenderConfig) -> None:
    print("\n" + "=" * 50)
    print("渲染性能测试结果")
    print(
        f"测试模式: Camera={render_config['render_camera']}, "
        f"Lidar={render_config['render_lidar']}"
    )
    print("=" * 50)
    print(f"总帧数: {stats['total_frames']}")
    print(f"平均耗时: {stats['mean_ms']:.2f}ms")
    print(f"中位数耗时: {stats['median_ms']:.2f}ms")
    print(f"最快耗时: {stats['min_ms']:.2f}ms")
    print(f"最慢耗时: {stats['max_ms']:.2f}ms")
    print(f"标准差: {stats['std_ms']:.2f}ms")
    print(f"平均FPS: {stats['fps']:.1f}")
    print("=" * 50)


def print_init_summary(
    config: BenchmarkConfig, render_config: RenderConfig, init_params: InitParams
) -> None:
    if render_config["render_camera"]:
        if config.get("sensors_json"):
            print(
                f"- 相机: {len(init_params.cameras)}个 (from {config['sensors_json']})"
            )
            for cam in init_params.cameras:
                print(f"  - {cam.id}: {cam.width}x{cam.height}")
        else:
            print(f"- 相机: {config['camera_count']}个, 分辨率 {config['resolution']}")
    if render_config["render_lidar"]:
        if config.get("sensors_json"):
            print(
                f"- 激光雷达: {len(init_params.lidars)}个 (from {config['sensors_json']})"
            )
            for lidar in init_params.lidars or []:
                print(
                    f"  - {lidar.id}: tile={lidar.tile_width}x{lidar.tile_height}, "
                    f"az_res={lidar.azimuth_resolution}, "
                    f"elev_ch={lidar.n_elevation_channels}"
                )
        else:
            print(f"- 激光雷达: {config['lidar_count']}个")
