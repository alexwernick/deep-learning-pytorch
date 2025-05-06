'''
1. Create model
2. Choose a loss function
3. Define a dataset
4. Set an optimizer
5. Run a training loop
  - Calculate loss (forward pass)
  - Compute gradients (backpropagation)
  - Updating model parameters

We use salary dataset
Since target is salary, hence continuous, this is a regression problem
As regression we use linear layer as the final output
Also regression specific loss function - mean squared error
'''

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

salaries = pd.read_csv('salary.csv')

# Define input features
features = salaries.iloc[:, 0:-1]

X = features.to_numpy()
print(X)

# Define target values 
target = salaries.iloc[:, -1]
y = target.to_numpy()
print(y)

# Instantiate dataset class
dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

# Create the model
model = nn.Sequential(nn.Linear(4, 2),
                      nn.Linear(2, 1))

# Create the loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# training loop
num_epochs = 4

for epoch in range(num_epochs):
    for data in dataloader:
        # Set gradients to zero
        # because it stores gradients from previous steps 
        # by default
        optimizer.zero_grad()
        feature, target = data
        # Run forward pass
        pred = model(feature)
        # Compute loss and gradients
        loss = criterion(pred, target.view(-1, 1))
        # compute gradients
        loss.backward()
        optimizer.step()