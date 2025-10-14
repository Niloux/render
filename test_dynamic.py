#!/usr/bin/env python3
"""简化版动态轨迹渲染脚本 - 单相机渲染并生成视频"""

import json
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from data_types import Camera, Vehicle
from render_manager import FrameParams, InitParams, RenderManager

# 环境配置
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# 简化配置
CONFIG = {
    "model_path": "/home/saimo/work/render/049_new.pth",
    "trajectory_json": "trajectory_data.json",
    "output_video": "trajectory_video.mp4",
    "resolution": (1600, 896),
    "fps": 30,
    "save_frames": False,  # 不保存单独帧，直接生成视频
}

# 标准相机参数（只用一个相机）
CAMERA_EXTRINSICS = [
    [6.12323400e-17, 4.99791693e-02, 9.98750260e-01, 1.89100000e00],
    [-1.00000000e00, 3.06034148e-18, 6.11558155e-17, 0.00000000e00],
    [0.00000000e00, -9.98750260e-01, 4.99791693e-02, 1.48500000e00],
    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
]

CAMERA_INTRINSICS = [
    [1.25281310e03, 0.00000000e00, 8.26588115e02],
    [0.00000000e00, 1.25281310e03, 4.69984663e02],
    [0.00000000e00, 0.00000000e00, 1.00000000e00],
]


def load_trajectory_data(json_path: str) -> List[dict]:
    """从JSON文件加载轨迹数据"""
    print(f"加载轨迹数据: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持 {"frames": [...]} 或直接 [...]
    frames = data.get("frames", data) if isinstance(data, dict) else data

    print(f"成功加载 {len(frames)} 帧数据")
    return frames


def create_single_camera() -> Camera:
    """创建单个相机"""
    return Camera("main_camera", CAMERA_EXTRINSICS, CAMERA_INTRINSICS, CONFIG["resolution"][0], CONFIG["resolution"][1])


def create_frame_params(frame_data: dict) -> FrameParams:
    """从JSON数据创建帧参数"""
    # 主车数据
    ego = frame_data.get("ego_vehicle", {})
    ego_pos = [ego.get("x", 0), ego.get("y", 0), ego.get("z", 0)]
    ego_yaw = ego.get("yaw", 0)

    # 环境车数据
    env_vehicles = []
    for vehicle_data in frame_data.get("environment_vehicles", []):
        vehicle = Vehicle(
            trajectory=[[vehicle_data.get("x", 0), vehicle_data.get("y", 0), vehicle_data.get("z", 0)]],
            yaw=vehicle_data.get("yaw", 0),
            type=vehicle_data.get("model_id", "obj_015"),
        )
        env_vehicles.append(vehicle)

    return FrameParams(
        ego_trajectory=ego_pos,
        ego_yaw=ego_yaw,
        env_vehicles=env_vehicles,
        timestamp=frame_data.get("timestamp", 20250912),
    )


def render_trajectory_to_video(render_manager: RenderManager, frames_data: List[dict]):
    """渲染轨迹并直接生成视频"""
    total_frames = len(frames_data)

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(CONFIG["output_video"], fourcc, CONFIG["fps"], CONFIG["resolution"])

    if not video_writer.isOpened():
        raise RuntimeError(f"无法创建视频文件: {CONFIG['output_video']}")

    print(f"开始渲染 {total_frames} 帧到视频...")

    try:
        for i, frame_data in enumerate(frames_data):
            # 创建帧参数
            frame_params = create_frame_params(frame_data)

            # 渲染帧
            t0 = time.perf_counter()
            frame_resp = render_manager.render_frame(frame_params)
            t1 = time.perf_counter()

            # 获取渲染图像（只有一个相机，取第一个）
            image: torch.Tensor = frame_resp.images["main_camera"]
            image = image.detach().cpu().numpy()

            # 确认范围在 [0,1]，转成 uint8
            if image.dtype != np.uint8:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            # 转换为 OpenCV 格式 (RGB → BGR)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # 写入视频
            video_writer.write(image_bgr)

            # 显示进度
            render_time = (t1 - t0) * 1000
            if (i + 1) % 10 == 0 or i == 0:
                progress = (i + 1) / total_frames * 100
                print(f"进度: {i + 1}/{total_frames} ({progress:.1f}%) - {render_time:.2f}ms/帧")

        print(f"视频渲染完成: {CONFIG['output_video']}")

    finally:
        video_writer.release()


def main():
    """主函数"""
    try:
        print("=" * 50)
        print("简化版动态轨迹渲染器（单相机）")
        print("=" * 50)

        # 检查文件
        if not Path(CONFIG["model_path"]).exists():
            raise FileNotFoundError(f"模型文件不存在: {CONFIG['model_path']}")

        if not Path(CONFIG["trajectory_json"]).exists():
            raise FileNotFoundError(f"轨迹文件不存在: {CONFIG['trajectory_json']}")

        # 加载轨迹数据
        frames_data = load_trajectory_data(CONFIG["trajectory_json"])

        # 初始化渲染器
        print("初始化渲染管理器...")
        render_manager = RenderManager(CONFIG["model_path"])

        # 创建单个相机
        camera = create_single_camera()
        init_params = InitParams([camera])

        # 初始化
        init_resp = render_manager.init(init_params)
        if not init_resp.init_status:
            raise RuntimeError("渲染器初始化失败")

        print("渲染器初始化成功")
        print(f"配置: 1个相机, {CONFIG['resolution']}分辨率, {len(frames_data)}帧")

        # 渲染并生成视频
        start_time = time.time()
        render_trajectory_to_video(render_manager, frames_data)
        total_time = time.time() - start_time

        print(f"\n完成! 总耗时: {total_time:.2f}秒")
        print(f"平均: {total_time / len(frames_data):.3f}秒/帧")
        print(f"输出视频: {CONFIG['output_video']}")

    except Exception as e:
        print(f"错误: {e}")
        raise


if __name__ == "__main__":
    main()
