import pathlib
from argparse import Namespace

import torch
from torch import nn
from tqdm.auto import trange


def get_configuration(**kwargs):
    defaults = {
        'batch_size': 4,
        'num_workers': 2,
        'device': torch.device('cpu'),
        'max_epochs': 100,
        'checkpoint_directory': pathlib.Path('.'),
        'save_every': 1,
    }
    
    return Namespace(**{**defaults, **kwargs})


def train_model(model, dataloader, config):
    config.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    model.train()
    
    train_loss = []
    
    epoch_progress = trange(config.max_epochs)
    for epoch in epoch_progress:
        epoch_progress.set_description(f'Epoch {epoch}')
        running_loss = 0.
        running_count = 0
        
        for inputs, outputs in dataloader:
            config.optimizer.zero_grad()
            
            pred_outputs = model(inputs.to(config.device))
            loss = config.criterion(outputs.to(config.device), pred_outputs)
            loss.backward()
            config.optimizer.step()
            
            running_loss += loss.item()
            running_count += inputs.size(0)
            
            epoch_progress.set_postfix({
                'train loss': f'{running_loss / running_count:.3f}'
            })
        
        train_loss.append(running_loss / running_count)
        torch.save(model.state_dict(), config.checkpoint_directory / f'model_checkpoint-{epoch}.pt')
        
    return train_loss


# custom loss function

class PatchlessLoss:

    def __init__(self, patch_box, loss_fn=nn.MSELoss()):
        self.patch_box = patch_box

        self.loss = loss_fn

    def __call__(self, targets, outputs):
        x_min, y_min, h, w = self.patch_box

        patch = torch.ones(1, 1, *outputs.shape[2:], dtype=torch.bool, device=outputs.device)
        patch[0, :, x_min:x_min+h, y_min:y_min+w] = 0
        self.patch = patch  # hack to save (last) patch

        return self.loss(targets * patch, outputs * patch)

