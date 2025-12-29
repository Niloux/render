import numpy as np
import open3d as o3d

DEFAULT_EXTRINSICS_SENSOR_T_WORLD = np.array(
    [
        [
            -4.588266398430291063e-03,
            -3.413667297520365119e-03,
            9.999836472098125872e-01,
            1.544154267170511075e00,
        ],
        [
            -9.999632354307952387e-01,
            -7.228375769820964344e-03,
            -4.612848415714690224e-03,
            -2.315740942895095494e-02,
        ],
        [
            7.244004295493749329e-03,
            -9.999680482191979358e-01,
            -3.380376081852865672e-03,
            2.115612062706179408e00,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def load_lidar_npy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """读取 lidar.npy 并返回 points(xyz) 与 intensity。"""
    arr = np.load(path)
    xyz = arr[:, :3].astype(np.float64, copy=False)
    intensity = arr[:, 3].astype(np.float64, copy=False)
    return xyz, intensity


def invert_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """将外参矩阵求逆（例如 sensorTworld -> worldTsensor）。"""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape != (4, 4):
        raise ValueError(f"extrinsics must be 4x4, got {extrinsics.shape}")
    return np.linalg.inv(extrinsics)


def apply_default_view_extrinsics(
    vis: o3d.visualization.Visualizer, sensor_t_world: np.ndarray
) -> None:
    """应用默认观测角度（输入为 sensorTworld，内部转换为 Open3D 需要的 worldTsensor）。"""
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = invert_extrinsics(sensor_t_world)
    view_control.convert_from_pinhole_camera_parameters(
        camera_params, allow_arbitrary=True
    )


def visualize_pointcloud(xyz: np.ndarray, intensity: np.ndarray) -> None:
    """用 Open3D 可视化点云，并用强度做灰度上色。"""
    valid = np.any(xyz != 0.0, axis=1)
    xyz = xyz[valid]
    intensity = intensity[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    vmin, vmax = np.percentile(intensity, [1, 99])
    denom = max(vmax - vmin, 1e-12)
    g = np.clip((intensity - vmin) / denom, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(np.stack([g, g, g], axis=1))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="lidar.npy", width=1280, height=720)
    vis.add_geometry(pcd)
    apply_default_view_extrinsics(vis, DEFAULT_EXTRINSICS_SENSOR_T_WORLD)
    vis.run()
    vis.destroy_window()


def main() -> None:
    """脚本入口：读取点云并启动可视化窗口。"""
    xyz, intensity = load_lidar_npy("/home/saimo/work/render/output/lidar1.npy")
    visualize_pointcloud(xyz, intensity)


if __name__ == "__main__":
    main()
