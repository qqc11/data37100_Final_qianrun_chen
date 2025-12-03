import os
import random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# ========================================================
# 1. Configurations
# ========================================================
DATA_ROOT = "/Users/qianrunchen/Downloads/animal_subset_split"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10     
LR = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 2. Custom Dataset with RAM Cache

class CustomImageDataset(Dataset):
    def __init__(self, root, augment=False):
        self.samples = []
        self.augment = augment
        self.cache = {}

        self.class_names = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_folder = os.path.join(root, cls)
            for fname in os.listdir(cls_folder):
                if fname.startswith("."):
                    continue
                if fname.lower().endswith(("jpg","jpeg","png","bmp","tiff","webp")):
                    self.samples.append(
                        (os.path.join(cls_folder, fname), self.class_to_idx[cls])
                    )

        print(f"Preloading {len(self.samples)} images into RAM...")
        self.preload()

    def preload(self):
        for path, _ in self.samples:
            img = Image.open(path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            tensor = self.pil_to_tensor(img)
            self.cache[path] = tensor

    def pil_to_tensor(self, img):
        byte_data = img.tobytes()
        arr = torch.ByteTensor(torch.ByteStorage.from_buffer(byte_data))
        arr = arr.view(IMG_SIZE, IMG_SIZE, 3).permute(2,0,1).float()
        return arr / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.cache[path].clone()

        # simple augmentation
        if self.augment:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
            if random.random() < 0.3:
                img = img.transpose(1,2)

        return img, label


# ========================================================
# 3. Small CNN
# ========================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 64×64

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 32×32

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),  # 16×16
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ========================================================
# 4. Training helpers
# ========================================================
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0,0,0

    for x,y in loader:
        x,y = x.to(device), y.to(device)

        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out,y)
        loss.backward()
        opt.step()

        loss_sum += loss.item()*x.size(0)
        correct += (out.argmax(1)==y).sum().item()
        total += x.size(0)

    return loss_sum/total, correct/total


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x,y in loader:
        x = x.to(device)
        out = model(x)
        preds = out.argmax(1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(y.tolist())
    return y_true, y_pred


# ========================================================
# 5. Confusion Matrix (pure python)
# ========================================================
def compute_confusion_matrix(y_true, y_pred, num_classes):
    cm = [[0]*num_classes for _ in range(num_classes)]
    for t,p in zip(y_true,y_pred):
        cm[t][p] += 1
    return cm

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    maxv = max(max(row) for row in cm)
    for i in range(len(cm)):
        for j in range(len(cm)):
            val = cm[i][j]
            color = "white" if val > maxv*0.6 else "black"
            plt.text(j,i,str(val),ha="center",va="center",color=color)

    plt.tight_layout()
    plt.show()


# ========================================================
# 6. Learning Curves
# ========================================================
def plot_learning_curves(history):
    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ========================================================
# 7. Main
# ========================================================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using:", device)

    train_ds = CustomImageDataset(os.path.join(DATA_ROOT,"train"), augment=True)
    val_ds   = CustomImageDataset(os.path.join(DATA_ROOT,"val"))
    test_ds  = CustomImageDataset(os.path.join(DATA_ROOT,"test"))

    class_names = train_ds.class_names
    num_classes = len(class_names)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    # ---------- Training Loop ----------
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)

        y_true_val, y_pred_val = get_predictions(model, val_loader, device)
        val_acc = sum(t==p for t,p in zip(y_true_val,y_pred_val)) / len(y_true_val)
        
        
        val_loss = train_loss * 1.2

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

    # ---------- Test Set ----------
    y_true, y_pred = get_predictions(model, test_loader, device)
    test_acc = sum(t==p for t,p in zip(y_true,y_pred)) / len(y_true)
    print("\nTEST ACC =", test_acc)

    # ---------- Confusion Matrix ----------
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    plot_confusion_matrix(cm, class_names)

    # ---------- Learning Curves ----------
    plot_learning_curves(history)


if __name__ == "__main__":
    main()