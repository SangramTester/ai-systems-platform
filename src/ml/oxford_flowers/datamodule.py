import urllib.request
import tarfile
import os
from typing import Optional
from torch.utils.data import random_split, Subset

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from monitored_dataset import MonitoredDataset

class OxfordFlowersDataModule:
    def __init__(self, data_dir: str, batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.full_dataset: Optional[MonitoredDataset] = None
        self.trainset: Optional[Dataset] = None
        self.testset: Optional[Dataset] = None
        self.valset: Optional[Dataset] = None

    def prepare_data(self):
      # Download the Oxford 102 Flowers dataset
      image_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
      labels_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"

      os.makedirs(self.data_dir, exist_ok=True)
      image_path = os.path.join(self.data_dir, "102flowers.tgz")
      labels_path = os.path.join(self.data_dir, "imagelabels.mat")

      # Download images if not already present
      if not os.path.exists(image_path):
        print(f"Downloading Oxford 102 Flowers images... {image_url}")
        urllib.request.urlretrieve(image_url, image_path)
        print(f"Images saved to {image_path}")
      else:
        print(f"Images already downloaded at {image_path}")

      # Extract images if jpg directory doesn't exist
      jpg_dir = os.path.join(self.data_dir, "jpg")
      if not os.path.exists(jpg_dir):
        print(f"Extracting {image_path}...")
        with tarfile.open(image_path, 'r:gz') as tar:
          tar.extractall(path=self.data_dir)
        print(f"Images extracted to {jpg_dir}")
      else:
        print(f"Images already extracted at {jpg_dir}")


      # Download labels if not already present
      if not os.path.exists(labels_path):
        print(f"Downloading Oxford 102 Flowers labels... {labels_url}")
        urllib.request.urlretrieve(labels_url, labels_path)
        print(f"Labels saved to {labels_path}")
      else:
        print(f"Labels already downloaded at {labels_path}")

    def setup(self, stage=None):
        base = MonitoredDataset(self.data_dir, transform=None)
        n = len(base)

        train_size = int(0.7 * n)
        val_size   = int(0.15 * n)
        test_size  = n - train_size - val_size

        g = torch.Generator().manual_seed(42)
        train_split, val_split, test_split = random_split(base, [train_size, val_size, test_size], generator=g)

        # Now rebuild datasets with transforms but re-use the split indices
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        trainset = MonitoredDataset(self.data_dir, transform=train_transform)
        valset   = MonitoredDataset(self.data_dir, transform=val_transform)
        testset  = MonitoredDataset(self.data_dir, transform=val_transform)

        self.trainset = Subset(trainset, train_split.indices)
        self.valset   = Subset(valset,   val_split.indices)
        self.testset  = Subset(testset,  test_split.indices)


    def train_dataloader(self):
      if self.trainset is None:                                                                                                                                                                           
        raise RuntimeError("Dataset not initialized. Call setup() first.")   
      return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
      if self.valset is None:                                                                                                                                                                           
        raise RuntimeError("Dataset not initialized. Call setup() first.")   
      return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
      if self.testset is None:                                                                                                                                                                           
        raise RuntimeError("Dataset not initialized. Call setup() first.")   
      return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)