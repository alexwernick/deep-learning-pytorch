from PIL import Image
from torch.utils.data import Dataset

class OmniglotDataset(Dataset):
    def __init__(self, transform, samples):
        self.transform = transform
        # Samples are tuples of size three
        # - the image location
        # - alphabet as a one hot vector 
        # - character class
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, alphabet, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        return img, alphabet, label
