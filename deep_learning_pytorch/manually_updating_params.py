import torch.nn as nn

# manually updating parameters
# Learning rate is typically small

model = nn.Sequential(nn.Linear(16, 8),
                     nn.Linear(8, 4),
                     nn.Linear(4, 2))
lr = 0.001

# Update the weights
weight = model[0].weight
weight_grad = model[0].weight.grad

weight = weight - lr * weight_grad

# Update the biases
bias = model[0].bias
bias_grad = model[0].bias.grad
bias = bias - lr * bias_grad