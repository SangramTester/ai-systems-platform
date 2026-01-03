import torch
import torch.nn as nn
import torch.nn.functional as F

class OxfordFlowersNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(64, 128, 3)
    self.conv3 = nn.Conv2d(128, 256, 3)
    self.conv4 = nn.Conv2d(256, 512, 3)
    self.conv5 = nn.Conv2d(512, 1024, 3)
    self.global_pool = nn.AdaptiveAvgPool2d(1)  # Reduces any HxW to 1x1
    self.fc1 = nn.Linear(1024, 128)
    self.dropout = nn.Dropout(0.5)  # Drop 50% of neurons during training
    self.fc3 = nn.Linear(128, 102)

    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(256)
    self.bn4 = nn.BatchNorm2d(512)
    self.bn5 = nn.BatchNorm2d(1024)
    
  def forward(self, x):
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    x = self.pool(F.relu(self.bn3(self.conv3(x))))
    x = self.pool(F.relu(self.bn4(self.conv4(x))))
    x = self.pool(F.relu(self.bn5(self.conv5(x))))
    x = self.global_pool(x)  # [batch, 256, 26, 26] → [batch, 256, 1, 1]
    x = torch.flatten(x, 1)  # [batch, 256, 1, 1] → [batch, 256]
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc3(x)

    return x


