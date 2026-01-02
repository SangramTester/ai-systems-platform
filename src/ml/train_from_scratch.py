from model import Net
import torch
import torch.optim as optim
from dataset import CIFAR10DataModule
import torch.nn as nn
import torchvision
from image_show import imshow
import argparse
import time

def train_one_epoch(model, dataloader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)

      optimizer.zero_grad()

      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)
      preds = outputs.argmax(dim=1)
      correct += (preds ==labels).sum().item()
      total += labels.size(0)

  if total == 0:
    return 0.0, 0.0
  return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0 
  total = 0

  for i, (data, labels) in enumerate(dataloader):
    data, labels = data.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(data)
        loss = criterion(outputs, labels)

    running_loss += loss.item() * data.size(0)
    preds = outputs.argmax(dim=1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)

  if total == 0:
    return 0.0, 0.0

  return running_loss / total, correct / total

def test(test_loader):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = Net().to(device)
  model.load_state_dict(torch.load("./cifar10_model.pt", weights_only=True))
  correct = 0
  total = 0
  correct_pred = { classname: 0 for classname in test_loader.dataset.classes }
  total_pred = { classname: 0 for classname in test_loader.dataset.classes }

  with torch.no_grad():
    for data in test_loader:
       images,labels = data
       images, labels = images.to(device), labels.to(device) 
       outputs = model(images)
       _, predictions = torch.max(outputs, 1)
       total += labels.size(0)
       correct += (predictions == labels).sum().item()
       for label, prediction in zip(labels, predictions):
          if label == prediction:
             correct_pred[test_loader.dataset.classes[label]] += 1
          total_pred[test_loader.dataset.classes[label]] += 1

  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
  for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname} is {accuracy:.1f} %')

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'], help='Mode: train or test')

  args = parser.parse_args()

  batch_size = 32
  epochs = 10
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # device = torch.device("cpu")  # Force CPU for consistent testing environment

  dm = CIFAR10DataModule(data_dir='./data', batch_size=batch_size)
  dm.prepare_data()
  dm.setup(stage='None')

  train_loader = dm.train_dataloader()
  test_loader = dm.test_dataloader()

  if args.mode == 'train':
    start_time = time.time()
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(1, epochs + 1):
      train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
      test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

      print(f'Epoch {epoch}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
      print(f'Epoch {epoch}/{epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
    
    end_time = time.time()
    print(f'Training completed on device: {device} in: {end_time - start_time:.2f} seconds')
    torch.save(model.state_dict(), "./cifar10_model.pt")
  else:
    test(test_loader)

    
if __name__ == "__main__":
    main()