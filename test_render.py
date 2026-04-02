#!/usr/bin/env python3
"""渲染性能测试脚本"""

import argparse
import os
import statistics
import time
from pathlib import Path
from typing import List, Tuple, TypedDict

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

# ================= 辅助函数 =================


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


def create_test_lidars(count: int) -> List[Lidar]:
    """创建测试激光雷达列表"""
    return [
        Lidar(
            f"lidar{i + 1}",
            STANDARD_EXTRINSICS,
            azimuth_resolution=0.140625,
            min_azimuth=-180.0,
            max_azimuth=180.0,
            n_elevation_channels=64,
            min_elevation=-17.55,
            max_elevation=2.5,
            tile_width=32,
            tile_height=8,
        )
        for i in range(count)
    ]


def create_test_scenario(
    config: BenchmarkConfig, render_config: RenderConfig
) -> Tuple[InitParams, FrameParams]:
    """创建标准测试场景"""
    cameras: List[Camera] = []
    if render_config["render_camera"]:
        cameras = create_test_cameras(
            config["camera_count"], config["resolution"][0], config["resolution"][1]
        )

    lidars: List[Lidar] = []
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
        Vehicle([8130.21911375, 4680.84724959, 49.77610101], 0, "obj_015"),
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
                    save_path = os.path.join(config["output_dir"], f"{lidar_id}.npy")
                    np.save(save_path, lidar_data.cpu().numpy())

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
        default="/home/saimo/work/render/049_default.pth",
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
    parser.add_argument("--camera-count", type=int, default=1, help="相机数量")
    parser.add_argument("--lidar-count", type=int, default=1, help="激光雷达数量")
    parser.add_argument("--width", type=int, default=1920, help="相机宽度")
    parser.add_argument("--height", type=int, default=1280, help="相机高度")

    args = parser.parse_args()

    render_config: RenderConfig = {
        "render_camera": not args.no_camera,
        # "render_lidar": not args.no_lidar,
        "render_lidar": False,
    }

    benchmark_config: BenchmarkConfig = {
        "model_path": args.model_path,
        "warmup_frames": args.warmup,
        "benchmark_frames": args.frames,
        "save_first_frame": not args.no_save,
        "output_dir": args.output_dir,
        "camera_count": args.camera_count,
        "resolution": (args.width, args.height),
        "lidar_count": args.lidar_count,
    }

    return benchmark_config, render_config


@torch.no_grad()
def main() -> None:
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
            print(f"- 相机: {config['camera_count']}个, 分辨率 {config['resolution']}")
        if render_config["render_lidar"]:
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
