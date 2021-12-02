import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import trange

from vit import HourglassViT

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_ds = CIFAR10('./data', download=True, transform=transform)
cifar_dl = DataLoader(cifar_ds, batch_size=4)

image_size = (480, 640)
patch_size = (16, 16)
model = HourglassViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=10,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1).float()


class SingleImageDataset(Dataset):
    
    def __init__(self, *, transform=None, std=0.1):
        self.transform = transform
        self.std = std
        
        self.image = plt.imread('data/grass.jpg').astype(np.float32)
        self.rand = np.random.uniform(0., 0.1, self.image.shape).astype(np.float32)

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        random_image = self.rand + np.random.randn(*self.rand.shape).astype(np.float32) * self.std
        transformed_image = self.image
        
        if self.transform:
            random_image = self.transform(random_image)
            transformed_image = self.transform(transformed_image)
        
        return random_image, transformed_image


ds = SingleImageDataset(transform=transforms.ToTensor())
dl = DataLoader(ds, batch_size=4, num_workers=4)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

model.train()

epoch_progress = trange(100)
for epoch in epoch_progress:
    
    running_loss = 0.
    running_count = 0
    
    for inputs, outputs in dl:
        optimizer.zero_grad()
        
        pred_outputs = model(inputs)
        loss = criterion(outputs, pred_outputs)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_count += inputs.size(0)
        
        epoch_progress.set_postfix({
            'train loss': f'{running_loss / running_count:.3}'
        })


