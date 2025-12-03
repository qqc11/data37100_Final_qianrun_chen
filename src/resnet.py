import os
import random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# 1. Basic Settings
# =========================
DATA_ROOT = "/Users/qianrunchen/Downloads/animal_subset_split"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 6
LR = 1e-4   # finetune
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

os.makedirs("results_resnet", exist_ok=True)

# =========================
# Device (Apple Silicon)
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# =========================
# Standard Transforms
# =========================
train_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

test_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =========================
# Dataset with correct unified class mapping
# =========================
class AnimalDataset(Dataset):
    def __init__(self, root, classes, transform):
        self.samples = []
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

        for c in classes:
            cls_dir = os.path.join(root, c)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(("jpg", "jpeg", "png", "bmp", "webp")):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[c]))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.samples)


# =========================
# ResNet18 Transfer Learning
# =========================
def create_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


    for name, param in model.named_parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =========================
# Training helpers
# =========================
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += loss_fn(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def get_preds(model, loader, device):
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


# =========================
# Plotting
# =========================
def plot_confusion(cm, classes, path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()


def plot_curves(history, prefix):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + "_acc.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(prefix + "_loss.png", dpi=200)
    plt.show()


# =========================
# Main
# =========================
def main():
    #  train
    class_names = sorted([
        d for d in os.listdir(os.path.join(DATA_ROOT, "train"))
        if os.path.isdir(os.path.join(DATA_ROOT, "train", d))
    ])
    print("Classes:", class_names)

    # Datasets with unified mapping
    train_ds = AnimalDataset(os.path.join(DATA_ROOT, "train"), class_names, train_tfm)
    val_ds   = AnimalDataset(os.path.join(DATA_ROOT, "val"),   class_names, test_tfm)
    test_ds  = AnimalDataset(os.path.join(DATA_ROOT, "test"),  class_names, test_tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Model
    model = create_resnet18(len(class_names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0
    best_state = None

    print("\n=== Training ResNet18 ===\n")

    for epoch in range(EPOCHS):
        tl, ta = train_epoch(model, train_loader, opt, loss_fn, device)
        vl, va = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        print(f"[Epoch {epoch+1}] Train Acc={ta:.4f} | Val Acc={va:.4f}")

        if va > best_acc:
            best_acc = va
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    # =========================
    # Final evaluation
    # =========================
    y_true, y_pred = get_preds(model, test_loader, device)
    print("\n=== Final Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, class_names, "results_resnet/confusion.png")
    plot_curves(history, "results_resnet/curve")


if __name__ == "__main__":
    main()
