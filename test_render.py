#!/usr/bin/env python3
"""渲染性能测试脚本 (整合版)"""

import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from data_types import Camera, Lidar, Vehicle
from render_manager import FrameParams, InitParams, RenderManager
from util import save_colors_as_png

# ================= 配置区域 =================

# 渲染模式配置 (在此处开关功能)
RENDER_CONFIG = {
    "render_camera": True,  # 是否测试相机渲染
    "render_lidar": True,  # 是否测试激光雷达渲染
}

# 基础测试配置
CONFIG = {
    "model_path": "/home/saimo/work/render/049_multimodal.pth",
    "warmup_frames": 1,
    "benchmark_frames": 100,
    "save_first_frame": True,  # 是否保存第一帧结果
    "output_dir": "output",  # 结果保存目录
    # 相机配置
    "camera_count": 1,
    "resolution": (1920, 1280),
    # 激光雷达配置
    "lidar_count": 1,
}

# 环境配置
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# ================= 标准参数 =================

STANDARD_EXTRINSICS = [
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

STANDARD_INTRINSICS = [
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


def create_test_scenario() -> Tuple[InitParams, FrameParams]:
    """创建标准测试场景"""
    cameras = []
    if RENDER_CONFIG["render_camera"]:
        cameras = create_test_cameras(CONFIG["camera_count"], *CONFIG["resolution"])

    lidars = []
    if RENDER_CONFIG["render_lidar"]:
        lidars = create_test_lidars(CONFIG["lidar_count"])

    init_params = InitParams(
        cameras=cameras,
        lidars=lidars,
        render_camera=RENDER_CONFIG["render_camera"],
        render_lidar=RENDER_CONFIG["render_lidar"],
    )

    # 创建测试车辆
    vehicles = [
        Vehicle([8235.21911375, 4684.84724959, 49.77610101], 0, "obj_015"),
    ]

    # 创建帧参数
    frame_params = FrameParams(
        ego_trajectory=[8212.159, 4684.092, 48.765],
        ego_yaw=0,
        env_vehicles=vehicles,
        timestamp=20250912,
    )

    return init_params, frame_params


def run_benchmark(render_manager: RenderManager, frame_params: FrameParams) -> Dict:
    """benchmark测试"""
    print("开始预热阶段...")
    for i in range(CONFIG["warmup_frames"]):
        render_manager.render_frame(frame_params)
        if i % 5 == 0:
            print(f"预热进度: {i + 1}/{CONFIG['warmup_frames']}")

    print("开始性能测试...")
    render_times = []

    # 确保输出目录存在
    if CONFIG["save_first_frame"]:
        os.makedirs(CONFIG["output_dir"], exist_ok=True)

    for i in range(CONFIG["benchmark_frames"]):
        t0 = time.perf_counter()
        frame_resp = render_manager.render_frame(frame_params)
        torch.cuda.synchronize()  # 确保GPU操作完成
        t1 = time.perf_counter()

        render_time = t1 - t0
        render_times.append(render_time)

        # 保存第一帧结果
        if i == 0 and CONFIG["save_first_frame"]:
            if RENDER_CONFIG["render_camera"] and frame_resp.images:
                print(f"正在保存相机图像到 {CONFIG['output_dir']}...")
                # 注意：util.save_colors_as_png 需要确保兼容性
                try:
                    save_colors_as_png(frame_resp.images)
                    # 如果需要移动到output目录，可以在这里添加移动逻辑
                except Exception as e:
                    print(f"保存相机图像失败: {e}")

            if RENDER_CONFIG["render_lidar"] and frame_resp.lidars:
                print(f"正在保存激光雷达点云到 {CONFIG['output_dir']}...")
                for lidar_id, lidar_data in frame_resp.lidars.items():
                    save_path = os.path.join(CONFIG["output_dir"], f"{lidar_id}.npy")
                    np.save(save_path, lidar_data.cpu().numpy())

            print("第一帧结果处理完成")

        if (i + 1) % 20 == 0:
            print(
                f"测试进度: {i + 1}/{CONFIG['benchmark_frames']} - 当前帧耗时: {render_time * 1000:.2f}ms"
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


def print_benchmark_results(stats: Dict) -> None:
    """打印格式化的benchmark结果"""
    print("\n" + "=" * 50)
    print("渲染性能测试结果")
    print(
        f"测试模式: Camera={RENDER_CONFIG['render_camera']}, Lidar={RENDER_CONFIG['render_lidar']}"  # noqa: E501
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


@torch.no_grad()
def main() -> None:
    """主测试函数"""
    try:
        model_path = Path(str(CONFIG["model_path"]))
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"初始化渲染管理器: {model_path}")
        render_manager = RenderManager(str(model_path))

        init_params, frame_params = create_test_scenario()

        print("初始化渲染器...")
        init_resp = render_manager.init(init_params)
        if not init_resp.init_status:
            raise RuntimeError("渲染器初始化失败")

        print("渲染器初始化成功")
        if RENDER_CONFIG["render_camera"]:
            print(f"- 相机: {CONFIG['camera_count']}个, 分辨率 {CONFIG['resolution']}")
        if RENDER_CONFIG["render_lidar"]:
            print(f"- 激光雷达: {CONFIG['lidar_count']}个")

        stats = run_benchmark(render_manager, frame_params)
        print_benchmark_results(stats)

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
