import torch


def load_and_inspect_pth(pth_path):  # noqa: C901
    """
    加载pth文件并检查其数据结构

    Args:
        pth_path (str): pth文件的路径
    """
    try:
        # 加载pth文件
        checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)

        print(f"=== PTH文件结构分析: {pth_path} ===")
        print(f"数据类型: {type(checkpoint)}")

        if isinstance(checkpoint, dict):
            print(f"\n字典键数量: {len(checkpoint)}")
            print("\n主要键名:")
            for key in checkpoint.keys():
                value = checkpoint[key]
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor {value.shape} ({value.dtype})")
                elif isinstance(value, dict):
                    print(f"  {key}: Dict (包含 {len(value)} 个子项)")
                elif isinstance(value, (int, float, str)):
                    print(f"  {key}: {type(value).__name__} = {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")

            # 详细检查每个键的内容
            print("\n=== 详细结构 ===")
            for key, value in checkpoint.items():
                print(f"\n[{key}]:")
                if isinstance(value, torch.Tensor):
                    print(f"  形状: {value.shape}")
                    print(f"  数据类型: {value.dtype}")
                    print(f"  设备: {value.device}")
                    if value.numel() < 10:
                        print(f"  值: {value}")
                elif isinstance(value, dict):
                    print(f"  子字典包含 {len(value)} 个键:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            print(f"    {sub_key}: Tensor {sub_value.shape}")
                        else:
                            print(f"    {sub_key}: {type(sub_value).__name__}")
                else:
                    print(f"  值: {value}")

        return checkpoint

    except Exception as e:
        print(f"加载失败: {e}")
        return None


if __name__ == "__main__":
    pth_path = "/home/saimo/work/render/debug.pth"

    checkpoint = load_and_inspect_pth(pth_path)
