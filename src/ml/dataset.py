import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
