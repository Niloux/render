import math
import os
import time

import numpy as np
import torch
from gsplat import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)
from PIL import Image

from models import GaussianComponent, GSModel

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GSModel.load_from_pth("model.pth").to_device(device)
background: GaussianComponent = model.get_component("background")

means = background.get_xyz()
quats = background.get_quats()
scales = background.get_scales()
opacities = background.get_opacities()
colors = background.get_colors()

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
]
Ks = [
    [[2049.873291015625, 0.0, 964.3667602539062], [0.0, 2049.873291015625, 644.5161743164062], [0.0, 0.0, 1.0]],
    [[2054.6259765625, 0.0, 967.3179321289062], [0.0, 2054.6259765625, 642.3396606445312], [0.0, 0.0, 1.0]],
    [[2046.391845703125, 0.0, 939.7431640625], [0.0, 2046.391845703125, 649.4197998046875], [0.0, 0.0, 1.0]],
]
img_width = 1920
img_height = 1280

viewmats = torch.tensor(viewmats, dtype=torch.float32, device=device)
Ks = torch.tensor(Ks, dtype=torch.float32, device=device)


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
        img = Image.fromarray(rgb_colors, "RGB")
        output_path = os.path.join(output_dir, f"view_{view_idx:02d}.png")
        img.save(output_path)
        print(f"保存视角 {view_idx} 到: {output_path}")

        # 保存统计信息
        print(f"  - 像素值范围: [{rgb_colors.min()}, {rgb_colors.max()}]")
        print(f"  - 平均像素值: {rgb_colors.mean():.3f}")


def extract_camera_centers(viewmats):
    """
    从view matrices提取相机中心位置

    Args:
        viewmats: 形状为 [C, 4, 4] 的view matrices

    Returns:
        camera_centers: 形状为 [C, 3] 的相机中心位置
    """
    R = viewmats[:, :3, :3]  # 旋转矩阵 [C, 3, 3]
    t = viewmats[:, :3, 3]  # 平移向量 [C, 3]
    # camera_center = -R^T * t
    camera_centers = -torch.bmm(R.transpose(-2, -1), t.unsqueeze(-1)).squeeze(-1)  # [C, 3]
    return camera_centers


def render_gaussian_splatting(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height):
    """
    执行高斯点云渲染的核心函数

    Args:
        means: 高斯点的3D位置 [N, 3]
        quats: 高斯点的四元数旋转 [N, 4]
        scales: 高斯点的缩放 [N, 3]
        opacities: 高斯点的不透明度 [N, 1]
        colors: 高斯点的球谐系数 [N, K, 3]
        viewmats: 视图矩阵 [C, 4, 4]
        Ks: 相机内参矩阵 [C, 3, 3]
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        render_colors: 渲染的颜色图像 [C, H, W, 4]
        render_alphas: 渲染的alpha通道 [C, H, W, 1]
    """
    # 投影
    project_results = fully_fused_projection(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmats,
        Ks=Ks,
        width=img_width,
        height=img_height,
        packed=False,
        near_plane=0.001,
        far_plane=1000,
        calc_compensations=True,
    )
    radii, means2d, depths, conics, compensations = project_results

    # 处理不透明度
    batch_opacities = opacities[None, :, 0].expand(viewmats.shape[0], -1)  # [C, N]
    if compensations is not None:
        batch_opacities = batch_opacities * compensations

    # Tile处理
    tile_width = math.ceil(img_width / float(16))
    tile_height = math.ceil(img_height / float(16))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, 16, tile_width, tile_height, packed=False, n_cameras=viewmats.shape[0]
    )
    isect_offsets = isect_offset_encode(isect_ids, viewmats.shape[0], tile_width, tile_height)

    # 球谐函数处理
    camera_centers = extract_camera_centers(viewmats)  # [C, 3]
    dirs = means[None, :, :] - camera_centers[:, None, :]  # [C, N, 3]
    masks = radii > 0  # [C, N]
    shs = colors.expand(viewmats.shape[0], -1, -1, -1)  # [C, N, K, 3]
    batch_colors = spherical_harmonics(1, dirs, shs, masks=masks)  # [C, N, 3]
    batch_colors = torch.clamp_min(batch_colors + 0.5, 0.0)
    batch_colors = torch.cat((batch_colors, depths[..., None]), dim=-1)

    # 光栅化
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        batch_colors,
        batch_opacities,
        img_width,
        img_height,
        16,
        isect_offsets,
        flatten_ids,
        backgrounds=None,
        packed=False,
        absgrad=True,
    )

    return render_colors, render_alphas


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
    render_gaussian_splatting(means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    for i in range(num_iterations):
        t0 = time.time()

        render_colors, render_alphas = render_gaussian_splatting(
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
    # 单次渲染测试
    print("=== 单次渲染测试 ===")
    t0 = time.time()
    render_colors, render_alphas = render_gaussian_splatting(
        means, quats, scales, opacities, colors, viewmats, Ks, img_width, img_height
    )
    t1 = time.time()
    print(f"单次渲染耗时: {t1 - t0:.6f} 秒")

    # 保存单次渲染结果
    print("\n保存单次渲染结果...")
    save_colors_as_png(render_colors)
    print("单次渲染结果保存完成！")

    # 性能基准测试
    print("\n=== 性能基准测试 ===")
    avg_time, times = benchmark_rendering(num_iterations=20, save_images=False)

    # 可选：更长时间的性能测试
    # print("\n=== 长时间性能测试 ===")
    # avg_time_long, times_long = benchmark_rendering(num_iterations=100, save_images=False)


if __name__ == "__main__":
    main()
