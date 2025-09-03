"""3DGS模型管理使用示例。"""

import torch

from models import GaussianComponent, GSModel


def main() -> None:
    """主函数演示如何使用GSModel。"""

    # 1. 从PTH文件加载模型
    print("=== 加载模型 ===")
    try:
        model = GSModel.load_from_pth("model.pth")
        print(f"成功加载模型: {model}")
        print(model.summary())
    except FileNotFoundError:
        print("model.pth文件不存在，创建示例模型")
        model = create_example_model()

    # 2. 获取模型信息
    print("\n=== 模型信息 ===")
    print(f"组件数量: {len(model)}")
    print(f"总点云数: {model.total_points:,}")
    print(f"内存使用: {model.total_memory:.2f} MB")
    print(f"组件名称: {model.component_names}")

    # 3. 按类型获取组件
    print("\n=== 按类型获取组件 ===")
    objects = model.get_components_by_type("obj")
    print(f"对象组件数量: {len(objects)}")

    background = model.get_component("background")
    if background:
        print(f"背景组件: {background}")

    sky = model.get_component("sky")
    if sky:
        print(f"天空组件: {sky}")

    # 4. 操作组件
    print("\n=== 组件操作 ===")

    # # 移除一个小的对象组件
    # smallest_obj = min(objects, key=lambda x: x.num_points) if objects else None
    # if smallest_obj:
    #     removed = model.remove_component(smallest_obj.name)
    #     print(f"移除组件: {removed.name} ({removed.num_points} 点)")
    #     print(f"移除后总点数: {model.total_points:,}")

    # 5. 设备转换
    print("\n=== 设备转换 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目标设备: {device}")

    if device.type == "cuda":
        gpu_model = model.to_device(device)
        print(f"GPU模型: {gpu_model}")
        print("成功转换到GPU")
    else:
        print("CUDA不可用，跳过GPU转换")

    # # 6. 保存模型
    # print("\n=== 保存模型 ===")
    # output_path = "modified_model.pth"
    # model.save_to_pth(output_path)
    # print(f"模型已保存到: {output_path}")


def create_example_model() -> GSModel:
    """创建一个示例模型用于演示。"""
    model = GSModel()
    model.iteration = 1000

    # 创建示例背景组件
    n_bg = 1000
    background = GaussianComponent(
        name="background",
        xyz=torch.randn(n_bg, 3),
        feature_dc=torch.randn(n_bg, 1, 3),
        feature_rest=torch.randn(n_bg, 3, 3),
        scaling=torch.randn(n_bg, 3),
        rotation=torch.randn(n_bg, 4),
        opacity=torch.randn(n_bg, 1),
        semantic=torch.empty(n_bg, 0),  # 空的语义信息
    )

    # 创建示例对象组件
    n_obj = 500
    obj_001 = GaussianComponent(
        name="obj_001",
        xyz=torch.randn(n_obj, 3),
        feature_dc=torch.randn(n_obj, 1, 3),
        feature_rest=torch.randn(n_obj, 3, 3),
        scaling=torch.randn(n_obj, 3),
        rotation=torch.randn(n_obj, 4),
        opacity=torch.randn(n_obj, 1),
        semantic=torch.empty(n_obj, 0),
    )

    # 创建示例天空组件
    n_sky = 800
    sky = GaussianComponent(
        name="sky",
        xyz=torch.randn(n_sky, 3),
        feature_dc=torch.randn(n_sky, 1, 3),
        feature_rest=torch.randn(n_sky, 3, 3),
        scaling=torch.randn(n_sky, 3),
        rotation=torch.randn(n_sky, 4),
        opacity=torch.randn(n_sky, 1),
        semantic=torch.randn(n_sky, 1),  # 天空有语义信息
    )

    # 添加组件到模型
    model.add_component(background)
    model.add_component(obj_001)
    model.add_component(sky)

    return model


if __name__ == "__main__":
    main()
