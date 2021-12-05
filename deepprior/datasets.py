import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class SingleImageDataset(Dataset):
    
    def __init__(self, filename, *, transform=None, target_transform=None, scale=0.1,
                 std=0.1):
        self.transform = transform
        self.target_transform = target_transform
        self.std = std
        
        self.image = plt.imread(filename).astype(np.float32)
        self.rand = np.random.uniform(0., scale, self.image.shape).astype(np.float32)
    
    def __len__(self):
        return 128  # even though single image, this number limits the batch size and sets the epoch size
    
    def __getitem__(self, idx):
        random_image = self.rand + np.random.randn(*self.rand.shape).astype(np.float32) * self.std
        transformed_image = self.image
        
        if self.transform:
            random_image = self.transform(random_image)
        if self.target_transform:
            transformed_image = self.target_transform(transformed_image)
        
        return random_image, transformed_image

    @property
    def image_size(self):
        return self.image.shape[:2]
