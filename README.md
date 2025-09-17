# 3D Gaussian Splatting 渲染系统

一个基于3D Gaussian Splatting技术的高性能实时渲染系统，支持多相机视角和动态场景渲染。

## 系统要求

- Python >= 3.12
- CUDA 兼容的 GPU
- PyTorch >= 2.8.0

## 安装

```bash
# 克隆项目
git clone <repository-url>
cd render

# 安装依赖
pip install -e .
```

## 核心概念

### 数据结构

- **Camera**: 定义相机参数（内参、外参、分辨率）
- **Vehicle**: 定义场景中的车辆对象（位置、朝向、类型）
- **InitParams**: 初始化参数（相机列表）
- **FrameParams**: 单帧渲染参数（自车状态、环境车辆、时间戳）

## 快速开始

### 1. 基础使用示例

```python
from data_types import Camera, Vehicle
from render_manager import RenderManager, InitParams, FrameParams

# 1. 创建渲染管理器
render_manager = RenderManager("path/to/model.pth")

# 2. 定义相机参数
cameras = [
    Camera(
        id="front_camera",
        extrinsics=[  # 4x4 外参矩阵
            [-9.703e-03, -1.072e-02, 9.999e-01, 1.539e00],
            [-9.999e-01, -4.840e-03, -9.756e-03, -2.432e-02],
            [4.944e-03, -9.999e-01, -1.067e-02, 2.115e00],
            [0.0, 0.0, 0.0, 1.0]
        ],
        intrinsics=[  # 3x3 内参矩阵
            [2049.873, 0.0, 964.367],
            [0.0, 2049.873, 644.516],
            [0.0, 0.0, 1.0]
        ],
        width=1920,
        height=1280
    )
]

# 3. 初始化渲染器
init_params = InitParams(cameras)
init_resp = render_manager.init(init_params)

if not init_resp.init_status:
    print(f"初始化失败: {init_resp.error_msg}")
    exit(1)

# 4. 定义场景
vehicles = [
    Vehicle([492.08, -147.71, -30.84], 1.728, "obj_034"),
    Vehicle([494.08, -149.71, -30.84], 1.728, "obj_016"),
]

frame_params = FrameParams(
    ego_trajectory=[498.28, -176.11, -31.95],  # 自车位置 (x, y, z)
    ego_yaw=1.728,                             # 自车朝向
    env_vehicles=vehicles,                     # 环境车辆
    timestamp=20250912                         # 时间戳
)

# 5. 渲染单帧
frame_resp = render_manager.render_frame(frame_params)

# 6. 保存结果
from util import save_colors_as_png
save_colors_as_png(frame_resp.images)
```

### 2. 多相机渲染

```python
# 创建多个相机
cameras = []
for i in range(6):
    camera = Camera(
        id=f"camera_{i}",
        extrinsics=get_camera_extrinsics(i),  # 你的相机外参
        intrinsics=get_camera_intrinsics(i),  # 你的相机内参
        width=1920,
        height=1280
    )
    cameras.append(camera)

# 初始化和渲染
init_params = InitParams(cameras)
render_manager.init(init_params)

# 渲染后会得到所有相机的图像
frame_resp = render_manager.render_frame(frame_params)
# frame_resp.images 是一个字典: {"camera_0": tensor, "camera_1": tensor, ...}
```

### 3. 性能测试

```python
import time
import statistics

def benchmark_rendering(render_manager, frame_params, num_frames=100):
    """性能测试函数"""
    
    # 预热
    for _ in range(10):
        render_manager.render_frame(frame_params)
    
    # 测试
    render_times = []
    for i in range(num_frames):
        start_time = time.perf_counter()
        frame_resp = render_manager.render_frame(frame_params)
        end_time = time.perf_counter()
        
        render_times.append(end_time - start_time)
        
        if (i + 1) % 20 == 0:
            print(f"进度: {i + 1}/{num_frames}")
    
    # 统计结果
    mean_time = statistics.mean(render_times) * 1000  # 转换为毫秒
    fps = 1.0 / statistics.mean(render_times)
    
    print(f"平均渲染时间: {mean_time:.2f}ms")
    print(f"平均FPS: {fps:.1f}")
    
    return render_times

# 使用示例
times = benchmark_rendering(render_manager, frame_params)
```

## API 参考

### RenderManager

主要的渲染管理类。

#### 构造函数

```python
RenderManager(model: str = "model.path")
```

- `model`: 预训练模型文件路径（.pth格式）

#### 方法

##### init(init_params: InitParams) -> InitResp

初始化渲染器。

**参数:**
- `init_params`: 包含相机列表的初始化参数

**返回:**
- `InitResp`: 包含初始化状态和错误信息

##### render_frame(frame_params: FrameParams) -> FrameResp

渲染单帧。

**参数:**
- `frame_params`: 包含场景状态的帧参数

**返回:**
- `FrameResp`: 包含渲染图像和时间戳

### 数据类型

#### Camera

```python
@dataclass
class Camera:
    id: str                           # 相机ID
    extrinsics: List[List[float]]     # 4x4外参矩阵
    intrinsics: List[List[float]]     # 3x3内参矩阵  
    width: int                        # 图像宽度
    height: int                       # 图像高度
```

#### Vehicle

```python
@dataclass
class Vehicle:
    trajectory: List[float]  # [x, y, z] 位置坐标
    yaw: float              # 朝向角度（弧度）
    type: str               # 车辆类型（如 "obj_034"）
```

#### FrameParams

```python
@dataclass
class FrameParams:
    ego_trajectory: List[float]  # [x, y, z] 自车位置
    ego_yaw: float              # 自车朝向（弧度）
    env_vehicles: List[Vehicle] # 环境车辆列表
    timestamp: int              # 时间戳
```

## 工具函数

### save_colors_as_png

```python
from util import save_colors_as_png

save_colors_as_png(images: Dict[str, torch.Tensor], output_dir="output")
```

将渲染结果保存为PNG图像文件。

**参数:**
- `images`: 相机ID到图像张量的映射
- `output_dir`: 输出目录路径

## 性能优化

### 内存管理

系统采用预分配buffer策略，避免每帧内存分配：

- 静态场景数据（背景、天空）预计算
- 动态数据（车辆）使用预分配的最大buffer
- 零内存分配的帧间渲染

### GPU优化

- 所有计算在GPU上进行
- 支持CUDA 12.0架构
- 异步内存传输优化

### 使用建议

1. **预热**: 首次渲染前进行10-20帧预热
2. **批处理**: 连续渲染多帧时重用RenderManager实例
3. **内存**: 避免频繁创建/销毁RenderManager对象

## 故障排除

### 常见问题

1. **模型文件不存在**
   ```
   FileNotFoundError: 模型文件不存在: /path/to/model.pth
   ```
   确保模型文件路径正确且文件存在。

2. **初始化失败**
   ```
   init_resp.init_status == False
   ```
   检查相机参数格式，确保外参为4x4矩阵，内参为3x3矩阵。

3. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   减少相机数量或降低分辨率。

4. **渲染结果异常**
   - 检查车辆类型名称是否在模型中存在
   - 验证坐标系统是否正确
   - 确认相机外参的坐标变换

### 调试模式

设置环境变量启用详细日志：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_LAUNCH_BLOCKING=1  # 同步CUDA调用以便调试
```

## 示例项目

完整的使用示例请参考 `test.py` 文件，包含：

- 标准相机配置
- 多车辆场景设置  
- 性能基准测试
- 结果保存和分析

运行示例：

```bash
python test.py
```
