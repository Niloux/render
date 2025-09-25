import os
import time
from typing import List

import numpy as np
import torch
from PIL import Image

from config import DEVICE, MAP_CENTER
from models import GaussianComponent, GSModel
from render_kernel import render

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

device = DEVICE
model = GSModel.load_from_pth(
    "/home/saimo/yyf/streetcrafter/output/waymo/waymo_val_049/trained_model_0916_1/iteration_240000.pth"
).to_device(device)
background: GaussianComponent = model.get_component("background")
sky: GaussianComponent = model.get_component("sky")

MAP_CENTER = torch.tensor(MAP_CENTER, device=device)


# 合并背景和天空的高斯点参数(静态)
# 计算每一帧的actor高斯点(动态)
# 合并静态点云和动态点云
means = torch.cat([background.get_xyz()])  # [N, 3]
quats = torch.cat([background.get_quats()])  # [N, 4]
scales = torch.cat([background.get_scales()])  # [N, 3]
opacities = torch.cat([background.get_opacities()])  # [N, 1]
colors = torch.cat([background.get_colors()])  # [N, 4, 3]

# from ply import load_gaussian_parameters_from_ply, save_gaussian_parameters_to_ply

# save_gaussian_parameters_to_ply(means, quats, scales, opacities, colors, "test.ply")

# means, quats, scales, opacities, colors = load_gaussian_parameters_from_ply("049_0925.ply", "cuda")

# 输入相机内外参和分辨率
extrinsics = [
    [
        [6.12323400e-17, 4.99791693e-02, 9.98750260e-01, 1.89100000e00],
        [-1.00000000e00, 3.06034148e-18, 6.11558155e-17, 0.00000000e00],
        [0.00000000e00, -9.98750260e-01, 4.99791693e-02, 1.48500000e00],
        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
]
intrinsics = [
    [
        [1.25281310e03, 0.00000000e00, 8.26588115e02],
        [0.00000000e00, 1.25281310e03, 4.69984663e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
]
img_width = 1600
img_height = 896

# 输入主车的轨迹点和航向角
ego_position = torch.tensor([8352.159, 4684.092, 48.765], device=device)
ego_position = ego_position - MAP_CENTER
ego_heading = -0.004


# 计算当前帧的viewmats
def create_viewmats(
    extrinsics: List[List[List[float]]], ego_heading: float, ego_position: torch.Tensor
) -> torch.Tensor:
    """
    根据相机外参、主车航向角和位置计算viewmats

    Args:
        extrinsics: 相机外参矩阵列表，每个元素为4x4的相机到车辆坐标系变换矩阵
        ego_heading: 主车航向角，单位为弧度
        ego_position: 主车在世界坐标系中的位置，已减去CENTER的torch.Tensor [x, y, z]

    Returns:
        viewmats: 世界坐标系到相机坐标系的变换矩阵，torch.Tensor类型，形状为[N, 4, 4]
    """
    # 计算旋转矩阵（绕Z轴旋转）
    cos_h = torch.cos(torch.tensor(ego_heading, device=ego_position.device))
    sin_h = torch.sin(torch.tensor(ego_heading, device=ego_position.device))

    # 构建车辆到世界坐标系的变换矩阵 (T_vehicle_to_world)
    ego_pose = torch.tensor(
        [
            [cos_h, -sin_h, 0.0, ego_position[0]],
            [sin_h, cos_h, 0.0, ego_position[1]],
            [0.0, 0.0, 1.0, ego_position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=ego_position.device,
        dtype=torch.float32,
    )

    viewmats_list = []

    for ext_matrix in extrinsics:
        # 将外参矩阵转换为torch.Tensor
        ext = torch.tensor(ext_matrix, device=ego_position.device, dtype=torch.float32)

        # 计算相机到世界坐标系的变换矩阵 c2w = T_vehicle_to_world @ T_camera_to_vehicle
        c2w = ego_pose @ ext

        # 计算世界坐标系到相机坐标系的变换矩阵 w2c = inv(c2w)
        w2c = torch.linalg.inv(c2w)

        viewmats_list.append(w2c)

    return torch.stack(viewmats_list)


# 创建缩放矩阵，仅缩放 f_x, f_y, c_x, c_y
scale = 1
img_width *= scale
img_height *= scale
scale_matrix = torch.tensor([[scale, 0, scale], [0, scale, scale], [0, 0, 1]], dtype=torch.float32, device=device)

# 使用函数计算viewmats和构造Ks
viewmats = create_viewmats(extrinsics, ego_heading, ego_position)
Ks = torch.tensor(intrinsics, dtype=torch.float32, device=device) * scale_matrix


def save_colors_as_png(colors_tensor, output_dir="output"):
    """
    将渲染的colors张量保存为PNG图像

    Args:
        colors_tensor: 形状为 [N_views, H, W, C] 的张量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 将张量移到CPU并转换为numpy
    colors_np = colors_tensor.detach().cpu().numpy()
    n_views, height, width, channels = colors_np.shape

    print(f"Colors张量形状: {colors_np.shape}")
    print(f"视角数量: {n_views}, 图像尺寸: {height}x{width}, 通道数: {channels}")

    for view_idx in range(n_views):
        # 获取单个视角的图像数据
        view_colors = colors_np[view_idx]  # [H, W, C]

        # 如果通道数大于3，取前3个通道作为RGB
        if channels > 3:
            rgb_colors = view_colors[:, :, :3]
            print(f"视角 {view_idx}: 使用前3个通道作为RGB")
        else:
            rgb_colors = view_colors

        # 数值范围处理：假设输出在[0,1]范围内，转换到[0,255]
        rgb_colors = np.clip(rgb_colors, 0, 1)
        rgb_colors = (rgb_colors * 255).astype(np.uint8)

        # 创建PIL图像并保存
        img = Image.fromarray(rgb_colors)
        output_path = os.path.join(output_dir, f"view_{view_idx:02d}.png")
        img.save(output_path)
        print(f"保存视角 {view_idx} 到: {output_path}")

        # 保存统计信息
        print(f"  - 像素值范围: [{rgb_colors.min()}, {rgb_colors.max()}]")
        print(f"  - 平均像素值: {rgb_colors.mean():.3f}")


def benchmark_rendering(num_iterations=10, save_images=False):
    """
    测试连续渲染的平均耗时

    Args:
        num_iterations: 测试迭代次数
        save_images: 是否保存最后一次渲染的图像

    Returns:
        avg_time: 平均渲染时间(秒)
        times: 所有渲染时间的列表
    """
    print(f"开始性能测试，共 {num_iterations} 次迭代...")

    times = []
    render_colors = None

    # 预热GPU
    print("GPU预热中...")
    viewmats = torch.tensor(
        [
            [
                [3.3247e-03, -9.9988e-01, -1.5147e-02, 9.2147e-01],
                [4.7917e-02, 1.5289e-02, -9.9873e-01, -3.2239e00],
                [9.9885e-01, 2.5947e-03, 4.7963e-02, -9.7571e01],
                [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
            ]
        ],
        device="cuda:0",
    )
    Ks = torch.tensor(
        [[[1.2528e03, 0.0000e00, 8.2659e02], [0.0000e00, 1.2528e03, 4.6998e02], [0.0000e00, 0.0000e00, 1.0000e00]]],
        device="cuda:0",
    )
    render(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height)
    # return
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    for i in range(num_iterations):
        t0 = time.time()

        render_colors, render_alphas = render(
            means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
        )
        print(f"{render_colors.shape=}")
        # 移除batch维度并转换到CPU
        img_data = render_colors[0].detach().cpu().numpy()  # [896, 1600, 4]

        rgb_colors = img_data[:, :, :3]  # 只取前3个通道 [H, W, 3]
        rgb_colors = np.clip(rgb_colors, 0, 1)
        rgb_colors = (rgb_colors * 255).astype(np.uint8)

        # 创建PIL图像并保存 - 不指定模式，让PIL自动推断
        output_path = "output.png"
        img = Image.fromarray(rgb_colors)
        img.save(output_path)
        print(f"Saved RGB image to {output_path}")

        quit()

        # 确保GPU计算完成
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.time()
        iteration_time = t1 - t0
        times.append(iteration_time)

        print(f"迭代 {i + 1:2d}/{num_iterations}: {iteration_time:.6f} 秒")

    avg_time = sum(times) / len(times)
    std_time = np.std(times)
    min_time = min(times)
    max_time = max(times)

    print("\n=== 性能测试结果 ===")
    print(f"平均渲染时间: {avg_time:.6f} 秒")
    print(f"标准差:       {std_time:.6f} 秒")
    print(f"最快时间:     {min_time:.6f} 秒")
    print(f"最慢时间:     {max_time:.6f} 秒")
    print(f"FPS (平均):   {1.0 / avg_time:.2f}")

    if save_images and render_colors is not None:
        print("\n保存最后一次渲染结果...")
        save_colors_as_png(render_colors)
        print("图像保存完成！")

    return avg_time, times


def main():
    """
    主函数：执行渲染测试和性能基准测试

    可以通过修改以下参数来自定义测试：
    - benchmark_rendering(num_iterations=N)：设置测试迭代次数
    - benchmark_rendering(save_images=True)：保存最后一次渲染的图像
    """

    # 性能基准测试
    print("\n=== 性能基准测试 ===")
    avg_time_long, times_long = benchmark_rendering(num_iterations=1, save_images=True)


if __name__ == "__main__":
    main()
