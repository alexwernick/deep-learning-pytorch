'''
- For non-convex functions, gradient descent is used
- PyTorch simplifies gradient descent with optimizers
- Stochastic gradient descent (SGD) is one such optimizer
'''

import torch.nn as nn
import torch.optim as optim


model = nn.Sequential(nn.Linear(16, 8),
                     nn.Linear(8, 4),
                     nn.Linear(4, 2))

# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Perform parameter updates
optimizer.step()