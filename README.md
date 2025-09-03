# 3DGS模型管理库

一个简洁实用的3D Gaussian Splatting (3DGS) 模型管理库，用于加载、操作和保存3DGS模型数据。

## 功能特性

- **简洁设计**: 只包含核心功能，避免过度设计
- **类型安全**: 完整的类型注解支持
- **内存管理**: 提供内存使用统计和设备转换
- **数据验证**: 自动验证tensor维度和数据一致性
- **组件管理**: 支持按类型管理不同的高斯点云组件

## 核心类

### GaussianComponent
表示单个高斯点云组件，包含：
- 位置坐标 (xyz)
- 特征信息 (feature_dc, feature_rest)
- 缩放和旋转参数
- 透明度和语义信息

### GSModel
主管理类，提供：
- 从PTH文件加载/保存模型
- 组件的增删改查操作
- 内存和统计信息
- 设备转换支持

## 使用示例

```python
from src import GSModel

# 加载模型
model = GSModel.load_from_pth("model.pth")

# 查看模型信息
print(model.summary())
print(f"总点数: {model.total_points:,}")
print(f"内存使用: {model.total_memory:.2f} MB")

# 获取特定组件
background = model.get_component("background")
objects = model.get_components_by_type("obj")

# 移除组件
model.remove_component("obj_001")

# 保存模型
model.save_to_pth("modified_model.pth")
```

## 运行示例

```bash
python example.py
```

## 依赖

- Python >= 3.12
- PyTorch >= 2.8.0

## 项目结构

```
src/
├── __init__.py          # 主模块导出
├── gaussian_component.py # 高斯组件数据类
└── gs_model.py          # 主管理类
example.py               # 使用示例
```