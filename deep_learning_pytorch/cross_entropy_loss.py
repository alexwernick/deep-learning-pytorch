from torch.nn import CrossEntropyLoss
import torch

# scores - model predictions before the final softmax function 
scores = torch.tensor([-5.2, 4.6, 0.8])
one_hot_target = torch.tensor([1, 0, 0])

criterion = CrossEntropyLoss()
print(criterion(scores.double(), one_hot_target.double()))

