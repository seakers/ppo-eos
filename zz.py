import torch

x = torch.tensor(0.129456789)

x = torch.round(x / .02) * .02

print(x)