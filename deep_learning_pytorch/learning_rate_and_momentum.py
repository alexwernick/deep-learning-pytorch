'''
- Training a neural network = solving an optimization problem


Optimizer Hyperparameters:
 
Learning Rate:
 - Controls the step size
 - Too high → poor performance
 - Too low → slow training
 - Typical range: 0.01 (10^-2) and 0.0001 (10^-4)

Momentum:
 - Controls the inertia
 - Helps escape local minimum
 - Too small → optimizer gets stuck
 - Typical range: 0.85 to 0.99
'''
import torch.optim as optim
import torch.nn as nn

model = nn.Sequential(nn.Linear(16, 8),
                     nn.Linear(8, 4),
                     nn.Linear(4, 2))

# learning rate: controls the step size
# momentum: adds inertia to avoid getting stuck in local minimum
sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
