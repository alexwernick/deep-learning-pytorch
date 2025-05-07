'''
Solution to unstable gradients
1. Proper weights initialization
2. Good activations
3. Batch normalization

1. Weight initialization
 - research shows that good initialization ensures:
   - variance of layer inputs = variance of layer outputs
   - variance of gradients the same before and after a layer
 - How to achieve this depends on activation function:
   - For ReLU and similar we can use He/Kaiming initialization
'''

import torch.nn.init as init

# He/Kaiming initialization
init.kaiming_uniform_(layer.weight)
print (layer.weight)