'''
- Overfitting happens when model does not generalize to unseen data
  - memorizes training data
  - Performs well on training data but poorly on validation data

- Possible causes:
Problem                     Solutions
-------                     ---------
Dataset is not large enough  Get more data / use data augmentation
Model has too much capacity  Reduce model size / add dropout
Weights are too large        Weight decay to force params to remain small


- 'Regularization' technique that randomly adds zeros to input tensor during training 
'''
import torch
import torch.nn as nn

# Regularization
model = nn.Sequential(nn.Linear(8, 4),
                     nn.ReLU(),
                     nn.Dropout(p=0.5))
features = torch.randn((1, 8))
print(model(features))


# Regularization with weight decay
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)
# • Controlled by the weight_decay parameter in the optimizer, typically set to a small value
#   (e.g., 0.0001)
# • Weight decay encourages smaller weights by adding a penalty during optimization
# • Helps reduce overfitting, keeping weights smaller and improving generalization

