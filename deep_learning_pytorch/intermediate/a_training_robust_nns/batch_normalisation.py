'''
After a layer
1. Normalise the later's outputs by:
  - Subtracting the mean
  - Dividing by the standard deviation
2. Scale and shift normalised outputs using learned parameters

Model learns optimal inputs distribution for each layer:
- Faster loss decrease
- Helps against unstable gradients
'''