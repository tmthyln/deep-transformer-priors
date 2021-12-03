import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class SingleImageDataset(Dataset):
    
    def __init__(self, *, transform=None, std=0.1):
        self.transform = transform
        self.std = std
        
        self.image = plt.imread('data/grass.jpg').astype(np.float32)
        self.rand = np.random.uniform(0., 0.1, self.image.shape).astype(np.float32)
    
    def __len__(self):
        return 128  # even though single image, this number limits the batch size
    
    def __getitem__(self, idx):
        random_image = self.rand + np.random.randn(*self.rand.shape).astype(np.float32) * self.std
        transformed_image = self.image
        
        if self.transform:
            random_image = self.transform(random_image)
            transformed_image = self.transform(transformed_image)
        
        return random_image, transformed_image
