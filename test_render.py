#!/usr/bin/env python3
"""渲染性能测试脚本"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, List, Tuple, TypedDict

import numpy as np
import torch

from data_types import Camera, Lidar, Vehicle
from render_manager import FrameParams, InitParams, RenderManager
from util import save_colors_as_png

# ================= 类型定义 =================


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


# ================= 标准参数 =================

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
    [
        0.000000000000000000e00,
        0.000000000000000000e00,
        0.000000000000000000e00,
        1.000000000000000000e00,
    ],
]

STANDARD_INTRINSICS: List[List[float]] = [
    [2.084604312956008926e03, 0.00000000e00, 9.334067577078354816e02],
    [0.00000000e00, 2.084604312956008926e03, 6.650223418347507049e02],
    [0.00000000e00, 0.00000000e00, 1.00000000e00],
]

LIDAR_EXTRINSICS: List[List[float]] = [
    [-0.852646995188958, -0.522486863163586, 0.000761194269898, 1.430000000000000],
    [0.522487398378445, -0.852645663278401, 0.001513746431570, 0.000000000000000],
    [-0.000141883631515, 0.001688405760096, 0.999998564576482, 2.184000000000000],
    [0, 0, 0, 1],
]

# ================= 辅助函数 =================


def _validate_matrix(mat: Any, rows: int, cols: int, name: str) -> List[List[float]]:
    """校验并规范化矩阵输入，确保为给定 rows x cols 的二维列表。"""
    if not isinstance(mat, list) or len(mat) != rows:
        raise ValueError(
            f"{name}期望为{rows}x{cols}二维列表，实际行数为{getattr(mat, '__len__', lambda: 'N/A')()}"  # noqa: E501
        )
    out: List[List[float]] = []
    for r in range(rows):
        row = mat[r]
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError(
                f"{name}第{r}行期望长度为{cols}，实际为{getattr(row, '__len__', lambda: 'N/A')()}"  # noqa: E501
            )
        out.append([float(x) for x in row])
    return out


def save_point_cloud_as_ply(
    save_path: str, point_cloud: torch.Tensor | np.ndarray
) -> None:  # noqa: E501
    if isinstance(point_cloud, torch.Tensor):
        points = point_cloud.detach().cpu().numpy()
    else:
        points = np.asarray(point_cloud)

    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError(
            f"point_cloud 期望形状为 [N,3] 或 [N,4]，实际为 {points.shape}"
        )  # noqa: E501

    points = points.astype(np.float32, copy=False)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    has_intensity = points.shape[1] == 4
    if has_intensity:
        dtype = np.dtype([
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
        ])  # noqa: E501
        packed = np.empty(points.shape[0], dtype=dtype)
        packed["x"] = points[:, 0]
        packed["y"] = points[:, 1]
        packed["z"] = points[:, 2]
        packed["intensity"] = points[:, 3]
        properties = (
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float intensity\n"
        )
    else:
        dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        packed = np.empty(points.shape[0], dtype=dtype)
        packed["x"] = points[:, 0]
        packed["y"] = points[:, 1]
        packed["z"] = points[:, 2]
        properties = "property float x\nproperty float y\nproperty float z\n"

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
    """将相机 specs 列表解析为 Camera 列表。"""
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("cameras 内容必须是列表")

    cameras: List[Camera] = []
    for idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"camera spec 必须是对象(dict)，实际为{type(spec)}")
        cam_id = str(spec.get("id", f"camera{idx + 1}"))
        extr = _validate_matrix(
            spec.get("extrinsics", STANDARD_EXTRINSICS), 4, 4, f"{cam_id}.extrinsics"
        )
        intr = _validate_matrix(
            spec.get("intrinsics", STANDARD_INTRINSICS), 3, 3, f"{cam_id}.intrinsics"
        )
        width = int(spec.get("width", default_width))
        height = int(spec.get("height", default_height))
        cameras.append(Camera(cam_id, extr, intr, width, height))

    return cameras


def load_lidars_from_specs(specs: Any) -> List[Lidar]:
    """将激光雷达 specs 列表解析为 Lidar 列表。"""
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("lidars 内容必须是列表")

    lidars: List[Lidar] = []
    for idx, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"lidar spec 必须是对象(dict)，实际为{type(spec)}")
        lidar_id = str(spec.get("id", f"lidar{idx + 1}"))
        extr = _validate_matrix(
            spec.get("extrinsics", LIDAR_EXTRINSICS), 4, 4, f"{lidar_id}.extrinsics"
        )

        lidars.append(
            Lidar(
                lidar_id,
                extr,
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
    """从单个 JSON 文件加载相机与激光雷达配置。"""
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


def load_cameras_from_json(
    path: str, default_width: int, default_height: int
) -> List[Camera]:
    """从 JSON 文件加载相机配置。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    specs: Any
    if isinstance(data, dict) and "cameras" in data:
        specs = data["cameras"]
    else:
        specs = data

    return load_cameras_from_specs(
        specs, default_width=default_width, default_height=default_height
    )


def create_test_cameras(count: int, width: int, height: int) -> List[Camera]:
    """创建测试相机列表"""
    return [
        Camera(
            f"camera{i + 1}",
            STANDARD_EXTRINSICS,
            STANDARD_INTRINSICS,
            width,
            height,
        )
        for i in range(count)
    ]


def load_lidars_from_json(path: str) -> List[Lidar]:
    """从 JSON 文件加载激光雷达配置。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    specs: Any
    if isinstance(data, dict) and "lidars" in data:
        specs = data["lidars"]
    else:
        specs = data

    return load_lidars_from_specs(specs)


def create_test_lidars(count: int) -> List[Lidar]:
    """创建测试激光雷达列表"""
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
    """创建标准测试场景"""
    cameras: List[Camera] = []
    lidars: List[Lidar] = []

    if config.get("sensors_json") and (
        render_config["render_camera"] or render_config["render_lidar"]
    ):  # noqa: E501
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
                config["camera_count"],
                config["resolution"][0],
                config["resolution"][1],
            )

        if render_config["render_lidar"]:
            lidars = create_test_lidars(config["lidar_count"])

    init_params = InitParams(
        cameras=cameras,
        lidars=lidars,
        render_camera=render_config["render_camera"],
        render_lidar=render_config["render_lidar"],
    )

    # 创建测试车辆
    vehicles = [
        Vehicle([8130.21911375, 4680.84724959, 49.77610101], 0, "obj_009"),
    ]

    # 创建帧参数
    frame_params = FrameParams(
        ego_trajectory=[8093.55, 4680.36, 48.68],
        ego_yaw=0.0242,
        env_vehicles=vehicles,
        timestamp=20260401,
    )

    return init_params, frame_params


def run_benchmark(  # noqa: C901
    render_manager: RenderManager,
    frame_params: FrameParams,
    config: BenchmarkConfig,
    render_config: RenderConfig,
) -> BenchmarkStats:
    """benchmark测试"""
    print("开始预热阶段...")
    for i in range(config["warmup_frames"]):
        render_manager.render_frame(frame_params)
        if i % 5 == 0:
            print(f"预热进度: {i + 1}/{config['warmup_frames']}")

    print("开始性能测试...")
    render_times: List[float] = []

    # 确保输出目录存在
    if config["save_first_frame"]:
        os.makedirs(config["output_dir"], exist_ok=True)

    for i in range(config["benchmark_frames"]):
        t0 = time.perf_counter()
        frame_resp = render_manager.render_frame(frame_params)
        torch.cuda.synchronize()  # 确保GPU操作完成
        t1 = time.perf_counter()

        render_time = t1 - t0
        render_times.append(render_time)

        # 保存第一帧结果
        if i == 0 and config["save_first_frame"]:
            if render_config["render_camera"] and frame_resp.images:
                print(f"正在保存相机图像到 {config['output_dir']}...")
                try:
                    save_colors_as_png(
                        frame_resp.images, output_dir=config["output_dir"]
                    )
                except Exception as e:
                    print(f"保存相机图像失败: {e}")

            if render_config["render_lidar"] and frame_resp.lidars:
                print(f"正在保存激光雷达点云到 {config['output_dir']}...")
                for lidar_id, lidar_data in frame_resp.lidars.items():
                    save_path = os.path.join(config["output_dir"], f"{lidar_id}.ply")
                    save_point_cloud_as_ply(save_path, lidar_data)

            print("第一帧结果处理完成")

        if (i + 1) % 20 == 0:
            print(
                f"测试进度: {i + 1}/{config['benchmark_frames']} - 当前帧耗时: {render_time * 1000:.2f}ms"  # noqa: E501
            )

    return {
        "total_frames": len(render_times),
        "mean_ms": statistics.mean(render_times) * 1000,
        "median_ms": statistics.median(render_times) * 1000,
        "min_ms": min(render_times) * 1000,
        "max_ms": max(render_times) * 1000,
        "std_ms": statistics.stdev(render_times) * 1000,
        "fps": 1.0 / statistics.mean(render_times),
    }


def print_benchmark_results(stats: BenchmarkStats, render_config: RenderConfig) -> None:
    """打印格式化的benchmark结果"""
    print("\n" + "=" * 50)
    print("渲染性能测试结果")
    print(
        f"测试模式: Camera={render_config['render_camera']}, Lidar={render_config['render_lidar']}"  # noqa: E501
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


def parse_args() -> Tuple[BenchmarkConfig, RenderConfig]:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="渲染性能测试脚本")

    # 基础配置
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/saimo/work/render/049_cnn_0421.pth",
        help="模型文件路径",
    )
    parser.add_argument("--warmup", type=int, default=1, help="预热帧数")
    parser.add_argument("--frames", type=int, default=100, help="测试帧数")
    parser.add_argument("--no-save", action="store_true", help="不保存第一帧结果")
    parser.add_argument("--output-dir", type=str, default="output", help="结果保存目录")

    # 渲染配置
    parser.add_argument("--no-camera", action="store_true", help="禁用相机渲染")
    parser.add_argument("--no-lidar", action="store_true", help="禁用激光雷达渲染")

    # 场景配置
    parser.add_argument(
        "--sensors-json",
        type=str,
        default="",
        help="相机+激光雷达配置JSON文件路径（包含 cameras/lidars）。提供该参数时将忽略 --camera-count/--lidar-count 的生成逻辑",  # noqa: E501
    )
    parser.add_argument("--camera-count", type=int, default=1, help="相机数量")
    parser.add_argument("--lidar-count", type=int, default=1, help="激光雷达数量")
    parser.add_argument("--width", type=int, default=1920, help="相机宽度")
    parser.add_argument("--height", type=int, default=1280, help="相机高度")

    args = parser.parse_args()

    render_config: RenderConfig = {
        "render_camera": not args.no_camera,
        "render_lidar": not args.no_lidar,
        # "render_lidar": False,
    }

    benchmark_config: BenchmarkConfig = {
        "model_path": args.model_path,
        "warmup_frames": args.warmup,
        "benchmark_frames": args.frames,
        "save_first_frame": not args.no_save,
        "output_dir": args.output_dir,
        "sensors_json": args.sensors_json,
        "camera_count": args.camera_count,
        "resolution": (args.width, args.height),
        "lidar_count": args.lidar_count,
    }

    return benchmark_config, render_config


@torch.no_grad()
def main() -> None:  # noqa: C901
    """主测试函数"""
    # 环境配置
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

    try:
        config, render_config = parse_args()

        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"初始化渲染管理器: {model_path}")
        render_manager = RenderManager(str(model_path))

        init_params, frame_params = create_test_scenario(config, render_config)

        print("初始化渲染器...")
        init_resp = render_manager.init(init_params)
        if not init_resp.init_status:
            raise RuntimeError("渲染器初始化失败")

        print("渲染器初始化成功")
        if render_config["render_camera"]:
            if config.get("sensors_json"):
                print(
                    f"- 相机: {len(init_params.cameras)}个 (from {config['sensors_json']})"  # noqa: E501
                )
                for cam in init_params.cameras:
                    print(f"  - {cam.id}: {cam.width}x{cam.height}")
            else:
                print(
                    f"- 相机: {config['camera_count']}个, 分辨率 {config['resolution']}"
                )
        if render_config["render_lidar"]:
            if config.get("sensors_json"):
                print(
                    f"- 激光雷达: {len(init_params.lidars)}个 (from {config['sensors_json']})"  # noqa: E501
                )
                for lidar in init_params.lidars:
                    print(
                        "  - "
                        f"{lidar.id}: tile={lidar.tile_width}x{lidar.tile_height}, "
                        f"az_res={lidar.azimuth_resolution}, elev_ch={lidar.n_elevation_channels}"  # noqa: E501
                    )
            else:
                print(f"- 激光雷达: {config['lidar_count']}个")

        stats = run_benchmark(render_manager, frame_params, config, render_config)
        print_benchmark_results(stats, render_config)

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
