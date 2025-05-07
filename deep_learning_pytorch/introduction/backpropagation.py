# Run a forward pass
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch

sample = torch.randn(1, 16)  # Batch size of 1, 16 features to match the input dimension
target = torch.tensor([1])  # Example class label (for CrossEntropyLoss, target should be class indices)

model = nn.Sequential(nn.Linear(16, 8),
                     nn.Linear(8, 4),
                     nn.Linear(4, 2))

# Run a forward pass
prediction = model(sample)

# Calculate the loss and gradient
criterion = CrossEntropyLoss()
loss = criterion(prediction, target)
loss.backward()

# Print the results
print(f"Prediction shape: {prediction.shape}")
print(f"Loss value: {loss.item()}")