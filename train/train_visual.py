# train_visual.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_fusion import RadarFusionDataset
from dataset import ra_augment   
from vision_head import VisionHeadLinear, VisionHeadMLP


CLASS_MAP = {
    'metal': 0,
    'glass': 1,
    'plastic': 2,
    'cardboard': 3,
}

TRAIN_SCENES = ['scene1','scene2']      
VAL_SCENES   = ['scene3']

ROOT_DIR = ""           
RGB_EMB_PATH = "minc_rgb_embeddings.npz"  

NUM_CLASSES = len(CLASS_MAP)


def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(batch_size=32):

    train_set = RadarFusionDataset(
        root_dir=ROOT_DIR,
        scene_list=TRAIN_SCENES,
        class_map=CLASS_MAP,
        rgb_emb_path=RGB_EMB_PATH,
        transform=None,
    )

    val_set = RadarFusionDataset(
        root_dir=ROOT_DIR,
        scene_list=VAL_SCENES,
        class_map=CLASS_MAP,
        rgb_emb_path=RGB_EMB_PATH,
        transform=None,
    )

    print(f"[Visual-only] Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for ra, y, vis_emb in loader:
        y = y.to(device).long()
        vis_emb = vis_emb.to(device)  

        optimizer.zero_grad()
        logits = model(vis_emb)        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * vis_emb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for ra, y, vis_emb in loader:
            y = y.to(device).long()
            vis_emb = vis_emb.to(device)

            logits = model(vis_emb)
            loss = criterion(logits, y)

            running_loss += loss.item() * vis_emb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    batch_size = 32
    num_epochs = 50
    lr = 1e-3
    weight_decay = 1e-4

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    # 2) 768 -> 128 -> 4 
    model = VisionHeadMLP(vis_dim=768, hidden_dim=128, num_classes=NUM_CLASSES, dropout=0.1).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_path = "vision_only_best.pth"

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  >> New best vision-only model saved to {best_path} (val_acc={best_val_acc:.3f})")

    print("Done. Best val_acc =", best_val_acc)


if __name__ == "__main__":
    main()
