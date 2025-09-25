#!/usr/bin/env python3
"""渲染性能测试脚本"""

import os
import statistics
import time
from pathlib import Path

from data_types import Camera, Vehicle
from render_manager import FrameParams, InitParams, RenderManager
from util import save_colors_as_png

# 环境配置
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# 测试配置
CONFIG = {
    "model_path": "/home/saimo/work/render/049.pth",
    "warmup_frames": 10,
    "benchmark_frames": 100,
    "save_first_frame": True,
    "camera_count": 6,
    "resolution": (1600, 896),
}

# 标准相机参数
STANDARD_EXTRINSICS = [
    [6.12323400e-17, 4.99791693e-02, 9.98750260e-01, 1.89100000e00],
    [-1.00000000e00, 3.06034148e-18, 6.11558155e-17, 0.00000000e00],
    [0.00000000e00, -9.98750260e-01, 4.99791693e-02, 1.48500000e00],
    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
]

STANDARD_INTRINSICS = [
    [1.25281310e03, 0.00000000e00, 8.26588115e02],
    [0.00000000e00, 1.25281310e03, 4.69984663e02],
    [0.00000000e00, 0.00000000e00, 1.00000000e00],
]


def create_test_cameras(count: int, width: int, height: int) -> list[Camera]:
    """创建测试相机列表"""
    return [Camera(f"camera{i + 1}", STANDARD_EXTRINSICS, STANDARD_INTRINSICS, width, height) for i in range(count)]


def create_test_scenario() -> tuple[InitParams, FrameParams]:
    """创建标准测试场景"""
    # 创建相机
    cameras = create_test_cameras(CONFIG["camera_count"], *CONFIG["resolution"])
    init_params = InitParams(cameras)

    # 创建测试车辆
    # 43, 15
    vehicles = [
        Vehicle([[8232.21911375, 4679.84724959, 49.77610101]], -0.016, "obj_015"),
        Vehicle([[8242.21911375, 4684.84724959, 49.77610101]], -0.016, "obj_015"),
        # Vehicle([494.07811834, -149.71372052, -30.84144724], 1.728, "obj_016"),
        # Vehicle([489.07811834, -145.71372052, -30.84144724], 1.728, "obj_010"),
    ]

    # 创建帧参数
    frame_params = FrameParams(
        ego_trajectory=[8352.159, 4684.092, 48.765], ego_yaw=0, env_vehicles=vehicles, timestamp=20250912
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
            save_colors_as_png(frame_resp.images)
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
