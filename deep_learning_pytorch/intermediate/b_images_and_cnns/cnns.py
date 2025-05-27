'''
Why not use linear layers for images?
  - Too many parameters to train
  - Don't recognise spatial patterns

Better to use a convolutional layer
 - Parameters are collected in smaller grids called filters
 - Slide filters of parameters over the input
 - At each position, perform convolution
 - Resulting feature map:
   - Preserves spatial patterns from input
   - Uses fewer parameters than linear layer
  - One filter = one feature map
  - Apply activations to feature map
  - All feature maps combined form the output
  - nn.Conv2d(3, 32, kernel_size=3)

To compute convolution: Compute dot product of input patch and filter (element wise multiplication)

We often add Zero-padding which adds zeros around the layers input
- Maintains spacial dimensions of the input and output tensors
- Ensures border pixels are treated equally to others

Max Pooling is another operation commonly used in convolutional layers
- Slide non-overlapping window over input
- At each position, retain only the maximum value
- Used after convolutional layers to reduce spatial dimensions

Calculate dims of classifier input:
- Input image dims = 3 x 64 x 64
- 1st convolution dims = 32 X 64 X 64 # increases feature maps
- max pooling dims = 32 x 32 x 32 # halves height and width
- 2nd convolution dims = 64 x 32 x 324 # increases feature maps
- max pooling dims = 64 x 16 x 16 # halves height and width

'''
from deep_learning_pytorch.intermediate.c_.cnns import Net

import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # 3 inputs corresponding to RGB
            # We use filters of 3x3 set by the kernel size argument
            # Zero-padding set to 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            # Max pooling halves feature map in height and width
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        # See notes on how the dims of this input was calculated above
        self.classifier = nn.Linear(64*16*16, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x