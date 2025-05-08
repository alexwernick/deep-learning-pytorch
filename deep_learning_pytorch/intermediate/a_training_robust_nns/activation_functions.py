'''
relu
- often used as the default activation function
- nn.functional.relu()
- problem: grad zero for negative inputs - dying neurons

elu
- nn.functional.relu()
- thanks to non-zero gradients for negative values helps against dying neurons
- Average output around zero - helps against vanishing grads

'''