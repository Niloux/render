import os

from data_types import Camera
from render_manager import FrameParams, InitParams, RenderManager, Vehicle
from util import save_colors_as_png

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"


def render_test():
    render_mngr = RenderManager("/home/saimo/work/render/model.pth")
    camera1 = Camera(
        "camera1",
        [
            [-9.703265255827278613e-03, -1.072251344945212778e-02, 9.998954317070867237e-01, 1.538897001763444461e00],
            [
                -9.999406983533636328e-01,
                -4.840233586415512712e-03,
                -9.755609433373464007e-03,
                -2.432485553238794215e-02,
            ],
            [4.944332104809027843e-03, -9.999307975275865124e-01, -1.067491151635933944e-02, 2.115484641063037685e00],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [[2049.873291015625, 0.0, 964.3667602539062], [0.0, 2049.873291015625, 644.5161743164062], [0.0, 0.0, 1.0]],
        1920,
        1280,
    )
    init_params = InitParams([camera1])
    init_resp = render_mngr.init(init_params)
    print(f"{init_resp=}")

    vehicle1 = Vehicle([492.07811834, -147.71372052, -30.84144724], 1.728, "obj_016")
    ego_position = [498.28, -186.11, -31.95]
    ego_heading = 1.728
    frame_params = FrameParams(ego_position, ego_heading, [vehicle1], 20250912)
    frame_resp = render_mngr.render_frame(frame_params)

    image = frame_resp.images["camera1"]
    print(f"{image.shape=}")

    save_colors_as_png(frame_resp.images)


if __name__ == "__main__":
    render_test()
