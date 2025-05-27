import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # define image processing layer
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128)
        )
        # define alphabet layer
        self.alphabet_layer = nn.Sequential(
            nn.Linear(30, 8),
            nn.ELU(),
        )
        # define classifier
        self.classifier = nn.Sequential(
            # input size is image layer output size + alphabet output count
            # output is character count (classifier)
            nn.Linear(128 + 8, 964),
        )

    def forward(self, x_image, x_alphabet):
        x_image = self.image_layer(x_image)
        x_alphabet = self.alphabet_layer(x_alphabet)
        x = torch.cat((x_image, x_alphabet), dim=1)
        return self.classifier(x)