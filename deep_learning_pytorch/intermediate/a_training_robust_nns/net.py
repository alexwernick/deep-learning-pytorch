import torch.nn as nn
import torch.nn.init as init

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 16)
        # see batch_normalisation.py for notes
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        # adding He/Kaiming initialization
        # see vanishing_and_exploding_gradients.py for justification
        init.kaiming_uniform_(self.fc1.weight)
        init.kaiming_uniform_(self.fc2.weight)
        init.kaiming_uniform_(
            self.fc3.weight,
            nonlinearity="sigmoid",
        )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        # see activation_functions.py for notes
        x = nn.functional.elu(x)
        x = nn.functional.elu(self.fc2(x))
        x = nn.functional.sigmoid(self.fc3(x))
        return x
        