import os
import time

import gsplat
import torch

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

viewmats = torch.tensor(viewmats, dtype=torch.float32, device=device)
Ks = torch.tensor(Ks, dtype=torch.float32, device=device)

with torch.no_grad():
    t0 = time.time()
    colors, alphas, meta = gsplat.rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 1920, 1280)
    t1 = time.time()
    print(f"渲染耗时: {t1 - t0:.6f} 秒")
