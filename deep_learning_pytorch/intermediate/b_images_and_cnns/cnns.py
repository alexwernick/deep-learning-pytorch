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
from deep_learning_pytorch.intermediate.c_sequences_and_rnns.lstm_and_gru import LSTMNet

import torch.nn as nn
import torch.optim as optim

num_epochs = 1
net = LSTMNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(
    net.parameters(), lr=0.001
)

for epoch in range(num_epochs):
    for seqs, labels in dataloader_train:
        seqs = seqs.view(32, 96, 1)
        outputs = net(seqs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# evaluation loop
import torch
from torchmetrics import MeanSquaredError
mse = MeanSquaredError()

net.eval()

with torch.no_grad():
    for seqs, labels in test_loader:
        seqs = seqs.view(32, 96, 1)
        outputs = net(seqs).squeeze()
        mse(outputs, labels)

print(f"Test MSE: {mse.compute()}")