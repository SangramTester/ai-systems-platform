
import torch
from datamodule import OxfordFlowersDataModule
from model import OxfordFlowersNet
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import json

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

    model = OxfordFlowersNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    assert train_dataloader is not None, "Train dataloader is None. Ensure that setup() has been called properly."

    # Lists to store metrics
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path("runs") / f"oxford_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_filename = run_dir / "metrics.csv"
    ckpt_filename = run_dir / "best_model.pt"


    # save config
    (run_dir / "config.json").write_text(json.dumps({
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": 0.001,
    }, indent=2))

    # Create log file with timestamp
    best_val_accuracy = 0
    
    with open(log_filename, "w", newline='') as f:
      writer = csv.writer(f)
      writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

      for epoch in range(epochs):
          train_loss, train_accuracy = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
          val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
          current_lr = optimizer.param_groups[0]['lr']
          
          # Store in memory
          history['epoch'].append(epoch + 1)
          history['train_loss'].append(train_loss)
          history['train_acc'].append(train_accuracy)
          history['val_loss'].append(val_loss)
          history['val_acc'].append(val_accuracy)
          history['lr'].append(current_lr)

          # Log to CSV
          writer.writerow([epoch+1, train_loss, train_accuracy, val_loss, val_accuracy, current_lr])
          f.flush()

          if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), ckpt_filename)
          
          # Print
          print(f"Epoch {epoch+1}/{epochs}")
          print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}")
          print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")
          print(f"  LR: {current_lr:.6f}, Best Val: {best_val_accuracy:.4f}")

    # Final plot
    plot_training_curves(run_dir, history)

    print(f"\nTraining complete! Logs saved to {log_filename}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

def plot_training_curves(run_dir, history):
    """Generate training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['epoch'], history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['epoch'], history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['epoch'], history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{run_dir}/training_curves.png", dpi=150)
    plt.close()
    print("  Saved training curves to training_curves.png")


if __name__ == "__main__":
    main()