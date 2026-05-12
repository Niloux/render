#!/usr/bin/env python3
"""渲染性能测试脚本。"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch

from benchmark_support import (
    BenchmarkConfig,
    RenderConfig,
    create_test_scenario,
    print_benchmark_results,
    print_init_summary,
    run_benchmark,
)


def parse_args() -> Tuple[BenchmarkConfig, RenderConfig]:
    parser = argparse.ArgumentParser(description="渲染性能测试脚本")
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
    parser.add_argument("--no-camera", action="store_true", help="禁用相机渲染")
    parser.add_argument("--no-lidar", action="store_true", help="禁用激光雷达渲染")
    parser.add_argument(
        "--sensors-json",
        type=str,
        default="",
        help="相机+激光雷达配置JSON文件路径，包含 cameras/lidars",
    )
    parser.add_argument("--camera-count", type=int, default=1, help="相机数量")
    parser.add_argument("--lidar-count", type=int, default=1, help="激光雷达数量")
    parser.add_argument("--width", type=int, default=1920, help="相机宽度")
    parser.add_argument("--height", type=int, default=1280, help="相机高度")
    args = parser.parse_args()

    render_config: RenderConfig = {
        "render_camera": not args.no_camera,
        "render_lidar": not args.no_lidar,
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
def main() -> None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
    try:
        config, render_config = parse_args()
        model_path = Path(config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        from render_manager import RenderManager

        print(f"初始化渲染管理器: {model_path}")
        render_manager = RenderManager(str(model_path))
        init_params, frame_params = create_test_scenario(config, render_config)

        print("初始化渲染器...")
        init_resp = render_manager.init(init_params)
        if not init_resp.init_status:
            raise RuntimeError("渲染器初始化失败")

        print("渲染器初始化成功")
        print_init_summary(config, render_config, init_params)
        stats = run_benchmark(render_manager, frame_params, config, render_config)
        print_benchmark_results(stats, render_config)
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as exc:
        print(f"测试失败: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
