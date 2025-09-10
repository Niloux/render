import torch

SKY_CENTER = [2.2004, 63.2183, 0.6995]
SKY_RADIUS = 111.5528

MAP_CENTER = [492.07811834, -147.71372052, -32.64144724]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
