'''
- As we have a regression style task we use Mean Squared Error Loss
  - mean squared error = avg((prediction - target)^2)
- Squaring ensures:
  - errors don't cancel out
  - penalizes large errors more
- In Pytorch criterion = nn.MSELoss()

Let's look at two concepts: (expanding and squeezing tensors)

Expanding
- Recurrent layers expect input shape (batch_size, seq_length, num_features)
- Our data input is size (32, 96). To make it (32, 96, 1) we use view
- seqs = seqs.view(32, 96, 1)

Squeezing tensors
- In evaluation loop we need to revert the reshaping done in the training loop
- Labels are of shape (batch_size)
- Model outputs are of shape (batch_size, 1)
- we can use squeeze
- out = net(seqs).squeeze()

'''
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from deep_learning_pytorch.intermediate.c_sequences_and_rnns.lstm_and_gru import LSTMNet


net = LSTMNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(
    net.parameters(), lr=0.001
)


# train
for epoch in range(num_epochs):
    for seqs, labels in dataloader_train:
        seqs = seqs.view(32, 96, 1)
        outputs = net(seqs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# evaluate
mse = torchmetrics.MeanSquaredError()

net.eval()
with torch.no_grad():
    for seqs, labels in test_loader:
        seqs = seqs.view(32, 96, 1)
        outputs = net(seqs).squeeze()
        mse(outputs, labels)

print(f"Test MSE: {mse.compute()}")