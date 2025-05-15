from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Define transforms
# - parse to tensor
# - resize to 128*128
train_transforms = transforms.Compose([
    # if you wanted to add data augmentation
    # to generate more training data you can add two lines below
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Resize((128, 128))
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

image, label = next(iter(dataloader_train))

# The shape is as follows: torch.Size([batch size, number of colour channels (3 RGB), image height, image width])
print(image.shape)

# To see image we need move dimensions around so height and width come before channels
image = image.squeeze().permute(1, 2, 0)
print(image.shape)

# To view
plt.imshow(image)
plt.show()