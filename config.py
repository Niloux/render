import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = device

# 049_1009
SKY_CENTER = torch.tensor([-129.8243, -5.7802, 3.6421], device=device)
SKY_RADIUS = torch.tensor(143.3051, device=device)

MAP_CENTER = torch.tensor([8254.21911375, 4682.84724959, 48.77610101], device=device)

# # 广汽版本
# SKY_CENTER = torch.tensor([94.9795, -1.1345, 2.8569], device=device)
# SKY_RADIUS = torch.tensor(279.4799, device=device)

# MAP_CENTER = torch.tensor([151.99728137, 3.17300073, -4.69054173], device=device)

# # 一汽版本
# SKY_CENTER = [13.0901, 8.4261, 4.1519]
# SKY_RADIUS = 324.5758

# MAP_CENTER = [-1.08300298e03, 3.57478483e03, 7.77111180e-03]

# # 049
# SKY_CENTER = [32.6423, 0.5757, 3.1557]
# SKY_RADIUS = 161.0890

# MAP_CENTER = [8254.21911375, 4682.84724959, 48.77610101]

# 121
# SKY_CENTER = [2.2004, 63.2183, 0.6995]
# SKY_RADIUS = 111.5528

# MAP_CENTER = [492.07811834, -147.71372052, -32.64144724]
