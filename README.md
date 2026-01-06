AI Systems Platform

This repository contains applied computer vision systems built in PyTorch, progressing from a baseline CIFAR-10 classifier to a custom CNN trained on the Oxford Flowers 102 dataset.

The focus is on:

correct PyTorch training workflows

clean dataset handling (no train/val leakage)

reproducible experiments

clear separation of data, model, and training logic

Projects Overview

1. CIFAR-10 (Baseline)

A simpler vision task used as a sanity check to validate:

data loading

training loop correctness

loss / accuracy behavior

model convergence

This serves as a warm-up and baseline before tackling a more complex dataset.

2. Oxford Flowers 102 (Main Project)

A custom convolutional neural network trained from scratch on the Oxford Flowers 102 dataset.

This is the primary project in the repository and demonstrates:

deeper CNN architectures

data augmentation

train/validation/test splitting

metric tracking over longer training runs

Dataset: Oxford Flowers 102

Dataset: Oxford Flowers 102

Images: 8,189 flower images across 102 classes

Source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

Split Strategy

Random split: 70% train / 15% validation / 15% test

Fixed seed: 42

⚠️ Note: This is not the official Oxford split. Results are therefore not directly comparable to published benchmarks, and this is documented intentionally for transparency.

Data Leakage Prevention

The dataset is split once using indices on an untransformed base dataset

Train / validation / test datasets reuse the same indices

Augmentations are applied only to the training split

Validation and test sets use deterministic transforms

Model Architecture (Oxford Flowers)

A custom CNN built from scratch using PyTorch:

5 convolutional blocks with increasing channels (64 → 1024)

Batch Normalization after each convolution

MaxPooling for spatial downsampling

Adaptive Average Pooling to remove dependency on input resolution

Fully connected head with dropout

Output layer with 102 logits (CrossEntropyLoss compatible)

This architecture avoids flattening large feature maps and keeps the classifier stable.

Training Setup

Framework: PyTorch

Loss: CrossEntropyLoss

Optimizer: Adam

Batch size: 64

Epochs: 50

Device: CPU / CUDA (auto-detected)

Reproducibility:

Fixed random seed

Deterministic dataset splits

Results
Oxford Flowers 102

Best validation accuracy: ~68%

Achieved using:

Random crop, horizontal flip, rotation, color jitter (train only)

Center crop for validation/test

Custom CNN trained from scratch

Accuracy is reported on the validation split defined above and reflects a non-official split.

How to Run
Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Train Oxford Flowers model
python src/ml/oxford_flowers/main.py

The script will:

download the dataset if not present

train for the configured number of epochs

report training and validation loss/accuracy per epoch

save the trained model weights to disk

Repository Structure
src/ml/
├── cifar10/ # Baseline vision task
├── oxford_flowers/ # Main project
│ ├── dataset.py
│ ├── datamodule.py
│ ├── model.py
│ └── main.py

Key Design Decisions

Custom CNN instead of pretrained backbone
Chosen to demonstrate understanding of CNN fundamentals and training dynamics.

AdaptiveAvgPool2d
Avoids hard-coding spatial dimensions and stabilizes the classifier head.

Explicit train / eval modes
Ensures correct behavior for BatchNorm and Dropout.

Augmentation only on training data
Prevents validation leakage and inflated metrics.

Known Limitations & Future Improvements

Switch to official Oxford102 splits for benchmark-comparable results

Add metric logging to disk (JSONL / CSV)

Save best checkpoint by validation accuracy

Try pretrained backbones (ResNet, EfficientNet) for higher accuracy

Add learning rate scheduling

Notes

This repository is intentionally structured as an evolving system, not a single tutorial or notebook.
Each project builds on the same training and evaluation principles, increasing in complexity.
