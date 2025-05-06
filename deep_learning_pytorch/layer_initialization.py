'''
- layer weights are initialized to small values
- keeping both input data and layer weights small ensure stable outputs
- In practice Transfer Learning is used which will reuse a model trained on a first task for a second similar task
'''
import torch
import torch.nn as nn

layer = nn.Linear(64, 128)

# uniform distribution
nn.init.uniform_(layer.weight)

print(layer.weight.min(), layer.weight.max())


# Example of how to save and load weights
layer = nn.Linear(64, 128)
torch.save(layer, 'layer.pth')

new_layer = torch.load('layer.pth')

