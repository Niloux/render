import os
from typing import List

from config import DEVICE
from models import GaussianComponent, GSModel

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

device = DEVICE
model1 = GSModel.load_from_pth("121.pth").to_device(device)
background: GaussianComponent = model1.get_component("background")
sky: GaussianComponent = model1.get_component("sky")
actors: List[GaussianComponent] = model1.get_components_by_type("obj")
names = [actor.name for actor in actors]
print(f"{names=}")  # 010,016,034是效果还ok的, 下标对应2, 5, 9

actor1: GaussianComponent = actors[1]
actor2: GaussianComponent = actors[5]
actor3: GaussianComponent = actors[9]


model2 = GSModel.load_from_pth("049_1020.pth").to_device(device)
model2.add_component(actor1)
model2.add_component(actor2)
model2.add_component(actor3)
model2.save_to_pth("049_1020_combined.pth")
