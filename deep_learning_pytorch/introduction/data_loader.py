import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

animals = pd.read_csv('animal_data.csv')

# Define input features
features = animals.iloc[:, 1:-1]

X = features.to_numpy()
print(X)

# Define target values 
target = animals.iloc[:, -1]
y = target.to_numpy()
print(y)

# Instantiate dataset class
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# access individual sample
input_sample, label_sample = dataset[0]
print('input sample:', input_sample)
print('label sample:', label_sample)


batch_size = 2
shuffle = 2

# Create dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

for batch_inputs, batch_labels in dataloader:
    print('batch inputs:', batch_inputs)
    print('batch labels:', batch_labels)