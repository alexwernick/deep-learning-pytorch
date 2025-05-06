import torch.nn.functional as F
import torch

print(F.one_hot(torch.tensor(0), num_classes = 3))

print(F.one_hot(torch.tensor(1), num_classes = 3))