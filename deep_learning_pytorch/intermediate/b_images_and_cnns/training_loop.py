from deep_learning_pytorch.intermediate.b_images_and_cnns.cnns import Net
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchmetrics import Recall, Precision

# Define transforms
# - parse to tensor
# - resize to 128*128
train_transforms = transforms.Compose([
    # if you wanted to add data augmentation
    # to generate more training data you can add two lines below
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])

dataset_train = ImageFolder(
    "clouds_train",
    transform=train_transforms,
)

dataset_test = ImageFolder(
    "clouds_test",
    transform=train_transforms,
)

# Create dataloader
dataloader_train = DataLoader(
    dataset_train,
    shuffle=True,
    batch_size=1
)

dataloader_test = DataLoader(dataset_test)

net = Net(num_classes=7)

# Training loop

# Loss used for multi class classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Iterate over epochs and training batches
for epoch in range(1):
    for batch_idx, (images, labels) in enumerate(dataloader_train):
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass: get model's outputs
        outputs = net(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Compute gradients
        loss.backward()
        
        # Optimizer's step: update params
        optimizer.step()
        print(f"Epoch [{epoch + 1}/2], Batch [{batch_idx + 1}/{len(dataloader_train)}], Loss: {loss.item():.4f}")

# Eval
'''
Precision and Recall: multi-class classification
In multi-class classification: separate precision and recall for each class
- Precision: Fraction of cumulus-predictions that were correct
- Recall: Fraction of all cumulus examples correctly predicted

Averaging multi-class metrics
- With 7 classes, we have 7 precision and 7 recall scores
- We can analyse them per-class, or aggregate:
  - Micro average: global calculation (imbalanced datasets)
  - Macro average: mean of per-class metrics (Care about performance on small classes)
  - Weighted average: weighted mean of per-class metrics (Consider errors in larger classes as more important)

'''

metric_precision = Precision(
    task="multiclass", num_classes=7, average="macro"
)
metric_recall = Recall(
    task="multiclass", num_classes=7, average="macro"
)

net.eval()

with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        metric_precision(preds, labels)
        metric_recall(preds, labels)

precision = metric_precision.compute()
recall = metric_recall.compute()

print(f"Precision: {precision}")
print(f"Recall: {recall}")