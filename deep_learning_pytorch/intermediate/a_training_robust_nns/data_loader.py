from deep_learning_pytorch.intermediate.a_training_robust_nns.water_dataset import WaterDataset
from torch.utils.data import DataLoader

dataset_train = WaterDataset('water_train.csv')

dataloader_train = DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True
)



