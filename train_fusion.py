import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model_fusion import RadarVisionFusionNet
from dataset_fusion import RadarFusionDataset   
from config import CLASS_MAP, TRAIN_SCENES, VAL_SCENES
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for ra, y, vis_emb in loader:
        ra = ra.to(device)             # [B,1,H,W]
        y = y.to(device).long()        # [B]
        vis_emb = vis_emb.to(device)   # [B,768]

        optimizer.zero_grad()
        logits = model(ra, vis_emb)    # [B,num_classes]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * ra.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, criterion, device, mode="full"):

    assert mode in ["full", "radar_only", "vision_only"]

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for ra, y, vis_emb in loader:
            ra = ra.to(device)             # [B,1,H,W]
            y = y.to(device).long()        # [B]
            vis_emb = vis_emb.to(device)   # [B,768]

            if mode == "radar_only":

                vis_emb = torch.zeros_like(vis_emb)
            elif mode == "vision_only":

                ra = torch.zeros_like(ra)

            logits = model(ra, vis_emb)
            loss = criterion(logits, y)

            running_loss += loss.item() * ra.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc



def main():
    root_dir = ""   # TODO: 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== Dataset / Dataloader ======
    train_set = RadarFusionDataset(root_dir, scene_list=TRAIN_SCENES, rgb_emb_path="minc_rgb_embeddings.npz")
    val_set   = RadarFusionDataset(root_dir,scene_list=VAL_SCENES, rgb_emb_path="minc_rgb_embeddings.npz")

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False)

    # ====== Model / Loss / Optim ======
    model = RadarVisionFusionNet(
        radar_ckpt_path="small_ra_cnn_best.pth",
        num_classes=4,          
        radar_feat_dim=32,
        vis_dim=768,
        hidden_dim=128,
        freeze_radar=True,      
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # train proj + classifier
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    log = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss_full": [],
        "val_acc_full": [],
    }
    for epoch in range(1, 51):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss_full, val_acc_full = evaluate(
            model, val_loader, criterion, device, mode="full"
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss(full)={val_loss_full:.4f}, val_acc(full)={val_acc_full:.3f}"
        )

        #  best model
        if val_acc_full > best_val_acc:
            best_val_acc = val_acc_full
            torch.save(model.state_dict(), "fusion_best.pth")
            print(f"  >> New best fusion model saved, val_acc={best_val_acc:.3f}")
            # ====== Record logs ======
        log["epoch"].append(epoch)
        log["train_loss"].append(train_loss)
        log["train_acc"].append(train_acc)
        log["val_loss_full"].append(val_loss_full)
        log["val_acc_full"].append(val_acc_full)

    print("Done. Best val_acc =", best_val_acc)
        # ====== Plot curves ======
    plt.figure()
    plt.plot(log["epoch"], log["train_loss"], label="train_loss")
    plt.plot(log["epoch"], log["val_loss_full"], label="val_loss(full)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("curve_loss.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(log["epoch"], log["train_acc"], label="train_acc")
    plt.plot(log["epoch"], log["val_acc_full"], label="val_acc(full)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("curve_acc.png", dpi=200)
    plt.close()

    # val_loss_full,   val_acc_full   = evaluate(model, val_loader, criterion, device, mode="full")
    # val_loss_radar,  val_acc_radar  = evaluate(model, val_loader, criterion, device, mode="radar_only")
    # val_loss_vision, val_acc_vision = evaluate(model, val_loader, criterion, device, mode="vision_only")

    # print(f"Full      : val_acc = {val_acc_full:.3f}")
    # print(f"Radar only: val_acc = {val_acc_radar:.3f}")
    # print(f"Vision only: val_acc = {val_acc_vision:.3f}")



if __name__ == "__main__":
    main()
