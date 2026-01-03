
import torch
import torchvision
from datamodule import OxfordFlowersDataModule
from model import OxfordFlowersNet

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
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    if total == 0:
      return 0.0, 0.0

    return running_loss / total, correct / total
        

def test(test_loader):
    pass  # Implementation of testing the model

def main():
    
    batch_size = 64
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = OxfordFlowersDataModule(data_dir="./data", batch_size=batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage=None)
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()


    # for images, labels in train_dataloader:
    #   print(f"First batch shape: {images.shape}")
    #   print(f"Pixel range: min={images.min():.2f}, max={images.max():.2f}")

    #   # Save first image to check augmentation
    #   torchvision.utils.save_image(images[0], "augmented_sample.png")
    #   print("Saved augmented_sample.png - check if it looks varied!")
    #   break


    model = OxfordFlowersNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    assert train_dataloader is not None, "Train dataloader is None. Ensure that setup() has been called properly."

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), "./oxford_model.pt")


if __name__ == "__main__":
    main()