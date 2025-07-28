import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns

# Config
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
MODEL_PATH = "models/weapon_resnet18.pth"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# Datasets & Loaders
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

classes = train_ds.classes
print("Classes:", classes)

# Model Setup
model = models.resnet18(pretrained=True)
in_feats = model.fc.in_features
model.fc = nn.Linear(in_feats, len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()*imgs.size(0)
        running_corrects += (preds==labels).sum().item()
        total += imgs.size(0)
    epoch_loss = running_loss/total
    epoch_acc  = running_corrects/total
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)

    # Validation
    model.eval()
    val_loss, val_corrects, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item()*imgs.size(0)
            val_corrects += (preds==labels).sum().item()
            val_total += imgs.size(0)
    val_loss /= val_total
    val_acc  = val_corrects/val_total
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch}/{EPOCHS}  "
          f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}  "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

# Save Model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved to", MODEL_PATH)

# Plot Metrics
epochs_range = range(1, EPOCHS+1)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.lineplot(epochs_range, train_acc_history, label="Train Acc")
sns.lineplot(epochs_range, val_acc_history,   label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()

plt.subplot(1,2,2)
sns.lineplot(epochs_range, train_loss_history, label="Train Loss")
sns.lineplot(epochs_range, val_loss_history,   label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()
