# train.py

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import RadarRADataset, ra_augment
from model import SmallRACNN
from config import CLASS_MAP, TRAIN_SCENES, VAL_SCENES, NUM_CLASSES


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(root_dir: str,
                    train_scenes=None,
                    val_scenes=None,
                    batch_size: int = 8):
    if train_scenes is None:
        train_scenes = TRAIN_SCENES
    if val_scenes is None:
        val_scenes = VAL_SCENES

    train_set = RadarRADataset(
        root_dir=root_dir,
        scene_list=train_scenes,
        class_map=CLASS_MAP,     
        transform=ra_augment,    
    )

    val_set = RadarRADataset(
        root_dir=root_dir,
        scene_list=val_scenes,
        class_map=CLASS_MAP,     
        transform=None,          
    )


    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)                  # [B, 1, H, W]
        y = y.to(device).long()           # [B]

        optimizer.zero_grad()
        logits = model(x)                 # [B, num_classes]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long()

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    root_dir = ""   
    num_classes = NUM_CLASSES        
    batch_size = 8
    num_epochs = 100
    lr = 1e-3
    weight_decay = 1e-4

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = get_dataloaders(
        root_dir=root_dir,
        train_scenes=TRAIN_SCENES,  
        val_scenes=VAL_SCENES,
        batch_size=batch_size
    )

    model = SmallRACNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_acc = 0.0
    best_model_path = "small_ra_cnn_best.pth"

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}"
        )


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  >> New best model saved to {best_model_path} (val_acc={best_val_acc:.3f})")

    print("Training finished. Best val_acc =", best_val_acc)


if __name__ == "__main__":
    main()
