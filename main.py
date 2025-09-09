import os
import time
from typing import List

import numpy as np
import torch
from PIL import Image

from models import GaussianComponent, GSModel
from render_kernel import render

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GSModel.load_from_pth("model.pth").to_device(device)
background: GaussianComponent = model.get_component("background")
sky: GaussianComponent = model.get_component("sky")
actors: List[GaussianComponent] = model.get_components_by_type("obj")
names = [actor.name for actor in actors]
print(f"{names=}")  # 010,016,034是效果还ok的, 下标对应2, 5, 9
actor: GaussianComponent = actors[9]

position = torch.tensor([0, 0, 1.8], device=device)
heading = 1.728

# 合并背景和天空的高斯点参数
means = torch.cat([background.get_xyz(), sky.get_xyz(), actor.get_xyz(heading=heading, position=position)])
quats = torch.cat([background.get_quats(), sky.get_quats(), actor.get_quats(heading=heading)])
scales = torch.cat([background.get_scales(), sky.get_scales(), actor.get_scales()])
opacities = torch.cat([background.get_opacities(), sky.get_opacities(), actor.get_opacities()])
colors = torch.cat([background.get_colors(), sky.get_colors(), actor.get_colors()])

viewmats = [
    [
        [0.9891819357872009, 0.14656783640384674, 0.006079603917896748, 2.466172456741333],
        [0.008798381313681602, -0.017908472567796707, -0.9998009204864502, 2.102562427520752],
        [-0.14642979204654694, 0.9890385270118713, -0.019004298374056816, 37.407142639160156],
        [0.0, 0.0, 0.0, 1.0],
    ],
    [
        [0.6000252366065979, 0.7999652028083801, 0.00503957737237215, 28.221078872680664],
        [0.022167669609189034, -0.010329311713576317, -0.9997009038925171, 2.345853567123413],
        [-0.7996739149093628, 0.5999574661254883, -0.023931210860610008, 24.904296875],
        [0.0, 0.0, 0.0, 1.0],
    ],
    [
        [0.8138973116874695, -0.5809189081192017, 0.010220356285572052, -24.17304039001465],
        [-0.004636919591575861, -0.024084700271487236, -0.9996991753578186, 1.9141963720321655],
        [0.5809902548789978, 0.8136050701141357, -0.022296147421002388, 28.57950210571289],
        [0.0, 0.0, 0.0, 1.0],
    ],
    [
        [0.98912947226739, 0.14696407952990476, 0.00494433210480903, -0.5148595529038485],
        [0.006459232615176294, -0.009832521033172998, -0.9999307975275867, 2.4055291183765446],
        [-0.1469052940028298, 0.9890929586535637, -0.010674911516359345, 37.37956928453708],
        [0.0, 0.0, 0.0, 1.0],
    ],
]
Ks = [
    [[2049.873291015625, 0.0, 964.3667602539062], [0.0, 2049.873291015625, 644.5161743164062], [0.0, 0.0, 1.0]],
    [[2054.6259765625, 0.0, 967.3179321289062], [0.0, 2054.6259765625, 642.3396606445312], [0.0, 0.0, 1.0]],
    [[2046.391845703125, 0.0, 939.7431640625], [0.0, 2046.391845703125, 649.4197998046875], [0.0, 0.0, 1.0]],
    [[2049.873291015625, 0.0, 964.3667602539062], [0.0, 2049.873291015625, 644.5161743164062], [0.0, 0.0, 1.0]],
]
scale = 1
img_width = 1920 * scale
img_height = 1280 * scale

# 创建缩放矩阵，仅缩放 f_x, f_y, c_x, c_y
scale_matrix = torch.tensor([[scale, 0, scale], [0, scale, scale], [0, 0, 1]], dtype=torch.float32, device=device)

viewmats = torch.tensor(viewmats, dtype=torch.float32, device=device)
Ks = torch.tensor(Ks, dtype=torch.float32, device=device) * scale_matrix


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
    render(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    for i in range(num_iterations):
        t0 = time.time()

        render_colors, render_alphas = render(
            means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
        )

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
    avg_time_long, times_long = benchmark_rendering(num_iterations=100, save_images=True)


if __name__ == "__main__":
    main()
