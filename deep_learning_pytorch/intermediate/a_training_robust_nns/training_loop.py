import torch
from deep_learning_pytorch.intermediate.a_training_robust_nns.water_dataset import WaterDataset
from deep_learning_pytorch.intermediate.a_training_robust_nns.net import Net
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy

dataset_train = WaterDataset('water_train.csv')

dataloader_train = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True
)

dataset_test = WaterDataset('water_test.csv')

dataloader_test = DataLoader(
    dataset_test
)

net = Net()

# Training loop
# Define loss function and optimizer
#   - BCELoss for binary classification
#   - SGD optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
# if use optim.Adagrad(net.parameters(), lr=0.01)
#  - will adapt lr for each param. decreases lr for params that are infrequently updated
#  - good for sparse data - data in which some features are not often observed
#  = may decrease lr too fast
# if we use optim.RMSprop(net.parameters(), lr=0.01)
#  - updates for each parameter based on size of previous gradients
# if we use optim.Adam(net.parameters(), lr=0.01)
#  - most versatile and widely used
#  - combines RMSprop with concept of momentum

# Iterate over epochs and training batches
for epoch in range(2):
    for batch_idx, (features, labels) in enumerate(dataloader_train):
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass: get model's outputs
        outputs = net(features.float())
        
        # Compute loss
        loss = criterion(
            outputs, labels.float().view(-1, 1)
        )
        
        # Compute gradients
        loss.backward()
        
        # Optimizer's step: update params
        optimizer.step()
        print(f"Epoch [{epoch + 1}/2], Batch [{batch_idx + 1}/{len(dataloader_train)}], Loss: {loss.item():.4f}")


# Model evaluation
# Set up accuracy metric
acc = Accuracy(task="binary")

# Put model in eval mode and iterate over test data batches with no gradients
net.eval()
with torch.no_grad():
    for features, labels in dataloader_test:
        # Pass data to model to get predicted probabilities
        outputs = net(features.float())
        # Compute predicted labels
        preds = (outputs >= 0.5).float()
        # Update accuracy metric
        acc(preds, labels.float().view(-1, 1))

accuracy = acc.compute()
print(f"Accuracy: {accuracy}")

