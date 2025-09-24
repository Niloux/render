import torch

SKY_CENTER = [32.6423, 0.5757, 3.1557]
SKY_RADIUS = 161.0890

MAP_CENTER = [8254.21911375, 4682.84724959, 48.77610101]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
