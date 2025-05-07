'''
Fine tuning is a type of Transfer Learning
- Smaller learning rate
- Train part and freeze part of Network
- Rule of thumb: freeze early layers of network and fine-tune layers closer to output layer
'''

import torch.nn as nn

model = nn.Sequential(nn.Linear(64, 128),
                     nn.Linear(128, 256))

for name, param in model.named_parameters():
    if name == '0.weight':
        param.requires_grad = False