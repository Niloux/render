#!/usr/bin/env python3
"""生成轨迹JSON数据文件"""

import json


def generate_trajectory_json():
    """生成轨迹数据并保存为JSON文件"""

    # 主车起始和终止坐标
    start_pos = (8182.4, 4682.2, 48.76)
    end_pos = (8352.2, 4684.1, 48.72)
    ego_yaw = -0.016  # 航向角保持不变

    # 帧数
    total_frames = 200

    # 环境车相对主车的偏移量
    # 环境车1: x+40, y-3, z+1
    env_car1_offset = (40.0, -3.0, 1.0)
    # 环境车2: x+50, y+2, z+1
    env_car2_offset = (50.0, 2.0, 1.0)

    frames_data = []

    print(f"生成 {total_frames} 帧轨迹数据...")
    print(f"主车: {start_pos} -> {end_pos}")
    print(f"环境车1偏移: {env_car1_offset}")
    print(f"环境车2偏移: {env_car2_offset}")

    for i in range(total_frames):
        # 计算插值比例 (0到1)
        t = i / (total_frames - 1) if total_frames > 1 else 0

        # 主车位置线性插值
        ego_x = start_pos[0] + t * (end_pos[0] - start_pos[0])
        ego_y = start_pos[1] + t * (end_pos[1] - start_pos[1])
        ego_z = start_pos[2] + t * (end_pos[2] - start_pos[2])

        # 环境车1位置 = 主车位置 + 偏移
        env1_x = ego_x + env_car1_offset[0]
        env1_y = ego_y + env_car1_offset[1]
        env1_z = ego_z + env_car1_offset[2]

        # 环境车2位置 = 主车位置 + 偏移
        env2_x = ego_x + env_car2_offset[0]
        env2_y = ego_y + env_car2_offset[1]
        env2_z = ego_z + env_car2_offset[2]

        # 创建帧数据
        frame_data = {
            "frame_id": i,
            "timestamp": 20250912 + i,  # 递增时间戳
            "ego_vehicle": {"x": round(ego_x, 3), "y": round(ego_y, 3), "z": round(ego_z, 3), "yaw": ego_yaw},
            "environment_vehicles": [
                {
                    "x": round(env1_x, 3),
                    "y": round(env1_y, 3),
                    "z": round(env1_z, 3),
                    "yaw": ego_yaw,  # 与主车相同
                    "model_id": "obj_015",
                },
                {
                    "x": round(env2_x, 3),
                    "y": round(env2_y, 3),
                    "z": round(env2_z, 3),
                    "yaw": ego_yaw,  # 与主车相同
                    "model_id": "obj_015",
                },
            ],
        }

        frames_data.append(frame_data)

        # 显示进度
        if (i + 1) % 50 == 0:
            print(f"生成进度: {i + 1}/{total_frames}")

    # 创建完整的JSON数据结构
    trajectory_data = {
        "metadata": {
            "total_frames": total_frames,
            "description": "自动生成的轨迹数据",
            "ego_start": start_pos,
            "ego_end": end_pos,
            "ego_yaw": ego_yaw,
            "env_vehicles": [
                {"id": 1, "offset": env_car1_offset, "model": "obj_015"},
                {"id": 2, "offset": env_car2_offset, "model": "obj_015"},
            ],
        },
        "frames": frames_data,
    }

    return trajectory_data


def save_trajectory_json(data, filename="trajectory_data.json"):
    """保存轨迹数据到JSON文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"轨迹数据已保存到: {filename}")


def print_trajectory_summary(data):
    """打印轨迹数据摘要"""
    frames = data["frames"]
    metadata = data["metadata"]

    print("\n" + "=" * 50)
    print("轨迹数据摘要")
    print("=" * 50)
    print(f"总帧数: {metadata['total_frames']}")
    print(f"主车起点: {metadata['ego_start']}")
    print(f"主车终点: {metadata['ego_end']}")
    print(f"主车航向角: {metadata['ego_yaw']}")
    print(f"环境车数量: {len(metadata['env_vehicles'])}")

    print("\n环境车配置:")
    for i, env_car in enumerate(metadata["env_vehicles"]):
        print(f"  车辆{i + 1}: 偏移{env_car['offset']}, 模型{env_car['model']}")

    # 显示前几帧和后几帧的数据
    print("\n前3帧数据:")
    for i in range(min(3, len(frames))):
        frame = frames[i]
        ego = frame["ego_vehicle"]
        print(f"  帧{i}: 主车({ego['x']:.1f}, {ego['y']:.1f}, {ego['z']:.2f})")

    print("\n后3帧数据:")
    for i in range(max(0, len(frames) - 3), len(frames)):
        frame = frames[i]
        ego = frame["ego_vehicle"]
        print(f"  帧{i}: 主车({ego['x']:.1f}, {ego['y']:.1f}, {ego['z']:.2f})")

    print("=" * 50)


def main():
    """主函数"""
    print("轨迹数据生成器")
    print("=" * 50)

    try:
        # 生成轨迹数据
        trajectory_data = generate_trajectory_json()

        # 打印摘要
        print_trajectory_summary(trajectory_data)

        # 保存文件
        save_trajectory_json(trajectory_data)

        print("\n生成完成!")

        # 验证数据
        print("\n数据验证:")
        first_frame = trajectory_data["frames"][0]
        last_frame = trajectory_data["frames"][-1]

        print(
            f"首帧主车位置: ({first_frame['ego_vehicle']['x']}, {first_frame['ego_vehicle']['y']}, {first_frame['ego_vehicle']['z']})"
        )
        print(
            f"末帧主车位置: ({last_frame['ego_vehicle']['x']}, {last_frame['ego_vehicle']['y']}, {last_frame['ego_vehicle']['z']})"
        )

        # 计算总移动距离
        dx = last_frame["ego_vehicle"]["x"] - first_frame["ego_vehicle"]["x"]
        dy = last_frame["ego_vehicle"]["y"] - first_frame["ego_vehicle"]["y"]
        dz = last_frame["ego_vehicle"]["z"] - first_frame["ego_vehicle"]["z"]
        total_distance = (dx**2 + dy**2 + dz**2) ** 0.5

        print(f"主车总移动距离: {total_distance:.2f}m")
        print(f"平均每帧移动: {total_distance / 199:.3f}m")  # 199个间隔

    except Exception as e:
        print(f"生成失败: {e}")
        raise


if __name__ == "__main__":
    main()
