from deep_learning_pytorch.intermediate.b_images_and_cnns.cnns import Net
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Define transforms
# - parse to tensor
# - resize to 128*128
train_transforms = transforms.Compose([
    # if you wanted to add data augmentation
    # to generate more training data you can add two lines below
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Resize((64, 64))
])

dataset_train = ImageFolder(
    "clouds_train",
    transform=train_transforms,
)

# Create dataloader
dataloader_train = DataLoader(
    dataset_train,
    shuffle=True,
    batch_size=1
)

net = Net(num_classes=7)

# Training loop

# Loss used for multi class classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)


# Iterate over epochs and training batches
for epoch in range(2):
    for images, labels in dataloader_train:
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


