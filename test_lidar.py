#!/usr/bin/env python3
"""渲染性能测试脚本"""

import os
import statistics
import time
from pathlib import Path

import torch

from data_types import Camera, Lidar, Vehicle
from render_manager import FrameParams, InitParams, RenderManager

# 环境配置
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# 测试配置
CONFIG = {
    "model_path": "/home/saimo/work/render/049_multimodal.pth",
    "warmup_frames": 1,
    "benchmark_frames": 100,
    "save_first_frame": True,
    "camera_count": 1,
    "resolution": (1920, 1280),
}

STANDARD_EXTRINSICS = [
    [-4.588266398430291063e-03, -3.413667297520365119e-03, 9.999836472098125872e-01, 1.544154267170511075e00],
    [-9.999632354307952387e-01, -7.228375769820964344e-03, -4.612848415714690224e-03, -2.315740942895095494e-02],
    [7.244004295493749329e-03, -9.999680482191979358e-01, -3.380376081852865672e-03, 2.115612062706179408e00],
    [0.0, 0.0, 0.0, 1.0],
]

STANDARD_INTRINSICS = [
    [2.084604312956008926e03, 0.00000000e00, 9.334067577078354816e02],
    [0.00000000e00, 2.084604312956008926e03, 6.650223418347507049e02],
    [0.00000000e00, 0.00000000e00, 1.00000000e00],
]


def create_test_cameras(count: int, width: int, height: int) -> list[Camera]:
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


def create_test_lidars(count: int) -> list[Lidar]:
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


def create_test_scenario() -> tuple[InitParams, FrameParams]:
    """创建标准测试场景"""
    # 创建相机
    cameras = create_test_cameras(CONFIG["camera_count"], *CONFIG["resolution"])
    lidars = create_test_lidars(1)
    init_params = InitParams(cameras, lidars, render_camera=False, render_lidar=True)

    # 创建测试车辆
    # 43, 15
    vehicles = [
        # Vehicle([-1070.146, 3554.094, 0.907585], 2.271827, "obj_006"),
        Vehicle([8235.21911375, 4684.84724959, 49.77610101], 0, "obj_015"),
    ]

    # 创建帧参数
    frame_params = FrameParams(
        ego_trajectory=[8212.159, 4684.092, 48.765],
        ego_yaw=0,
        env_vehicles=vehicles,
        timestamp=20250912,
        # ego_trajectory=[-1062.67011, 3550.622584, 0.013875],
        # ego_yaw=2.26268,
        # env_vehicles=vehicles,
        # timestamp=20250912,
    )

    return init_params, frame_params


def run_benchmark(render_manager: RenderManager, frame_params: FrameParams) -> dict:
    """benchmark测试

    返回详细的性能统计数据
    """
    print("开始预热阶段...")
    # 预热阶段 - 避免首次运行的初始化开销
    for i in range(CONFIG["warmup_frames"]):
        render_manager.render_frame(frame_params)
        if i % 5 == 0:
            print(f"预热进度: {i + 1}/{CONFIG['warmup_frames']}")

    print("开始性能测试...")
    # 正式测试阶段
    render_times = []

    for i in range(CONFIG["benchmark_frames"]):
        t0 = time.perf_counter()  # 使用高精度计时器
        frame_resp = render_manager.render_frame(frame_params)
        t1 = time.perf_counter()

        render_time = t1 - t0
        render_times.append(render_time)

        # 保存第一帧图像
        if i == 0 and CONFIG["save_first_frame"]:
            import numpy as np

            os.makedirs("output", exist_ok=True)
            lidar = frame_resp.lidars["lidar1"]
            np.save("output/lidar.npy", lidar.cpu().numpy())
            print("第一帧图像已保存")

        # 进度显示
        if (i + 1) % 20 == 0:
            print(f"测试进度: {i + 1}/{CONFIG['benchmark_frames']} - 当前帧耗时: {render_time * 1000:.2f}ms")

    # 统计分析
    return {
        "total_frames": len(render_times),
        "mean_ms": statistics.mean(render_times) * 1000,
        "median_ms": statistics.median(render_times) * 1000,
        "min_ms": min(render_times) * 1000,
        "max_ms": max(render_times) * 1000,
        "std_ms": statistics.stdev(render_times) * 1000,
        "fps": 1.0 / statistics.mean(render_times),
    }


def print_benchmark_results(stats: dict) -> None:
    """打印格式化的benchmark结果"""
    print("\n" + "=" * 50)
    print("渲染性能测试结果")
    print("=" * 50)
    print(f"总帧数: {stats['total_frames']}")
    print(f"平均耗时: {stats['mean_ms']:.2f}ms")
    print(f"中位数耗时: {stats['median_ms']:.2f}ms")
    print(f"最快耗时: {stats['min_ms']:.2f}ms")
    print(f"最慢耗时: {stats['max_ms']:.2f}ms")
    print(f"标准差: {stats['std_ms']:.2f}ms")
    print(f"平均FPS: {stats['fps']:.1f}")
    print("=" * 50)


@torch.no_grad()
def main() -> None:
    """主测试函数，包含错误处理和资源管理"""
    try:
        # 验证模型文件存在
        model_path = Path(CONFIG["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"初始化渲染管理器: {model_path}")
        render_manager = RenderManager(str(model_path))

        # 创建测试场景
        init_params, frame_params = create_test_scenario()

        # 初始化渲染器
        init_resp = render_manager.init(init_params)
        if not init_resp.init_status:
            raise RuntimeError("渲染器初始化失败")

        print("渲染器初始化成功")
        print(f"测试配置: {CONFIG['camera_count']}个相机, {CONFIG['resolution']}分辨率")

        # 运行benchmark
        stats = run_benchmark(render_manager, frame_params)

        # 显示结果
        print_benchmark_results(stats)

    except Exception as e:
        print(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    main()
