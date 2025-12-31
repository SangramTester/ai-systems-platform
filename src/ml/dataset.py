from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size


    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.trainset = torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)

    def classes(self):
      return ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    