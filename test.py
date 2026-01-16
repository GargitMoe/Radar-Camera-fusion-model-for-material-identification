# test_fusion.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RadarRADataset
from dataset_fusion import RadarFusionDataset
from model import SmallRACNN
from model_fusion import RadarVisionFusionNet
from vision_head import VisionHeadLinear, VisionHeadMLP

from train import set_seed
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """
    cm: numpy array (C × C)
    class_names: list of class names
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------
# Radar-only dataloader
# --------------------------------------------------------
def get_radar_test_loader(root_dir, scenes, class_map, batch_size=8):
    dataset = RadarRADataset(
        root_dir=root_dir,
        scene_list=scenes,
        class_map=class_map,
        transform=None
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


# --------------------------------------------------------
# Visual-only dataloader  (取 vis_emb 而不使用雷达)
# --------------------------------------------------------
def get_visual_test_loader(root_dir, scenes, class_map, rgb_emb_path, batch_size=8):
    dataset = RadarFusionDataset(
        root_dir=root_dir,
        scene_list=scenes,
        class_map=class_map,
        rgb_emb_path=rgb_emb_path,
        transform=None
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


# --------------------------------------------------------
# Fusion dataloader
# --------------------------------------------------------
def get_fusion_test_loader(root_dir, scenes, class_map, rgb_emb_path, batch_size=8):
    return get_visual_test_loader(root_dir, scenes, class_map, rgb_emb_path, batch_size)


# --------------------------------------------------------
# Radar-only evaluation
# --------------------------------------------------------
def test_radar_only(radar_ckpt, root_dir, scenes, class_map, num_classes, device):
    print("\n=== Radar-only Evaluation ===")

    loader = get_radar_test_loader(root_dir, scenes, class_map)
    model = SmallRACNN(num_classes=num_classes).to(device)

    state = torch.load(radar_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for ra, y in loader:
            ra = ra.to(device)
            y = y.to(device)

            logits = model(ra)
            pred = logits.argmax(dim=1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print(classification_report(all_labels, all_preds, digits=4))
    plot_confusion_matrix(cm,
                      class_names=list(class_map.keys()),
                      title="Radar-only Confusion Matrix")



# --------------------------------------------------------
# Visual-only evaluation
# --------------------------------------------------------
def test_visual_only(visual_ckpt, root_dir, scenes, class_map, rgb_emb_path, num_classes, device):
    print("\n=== Visual-only Evaluation ===")

    loader = get_visual_test_loader(root_dir, scenes, class_map, rgb_emb_path)
    
    # 你可以自由切换线性或 MLP 头：
    # model = VisionHeadLinear(vis_dim=768, num_classes=num_classes).to(device)
    model = VisionHeadMLP(vis_dim=768, hidden_dim=128, num_classes=num_classes).to(device)

    state = torch.load(visual_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for ra, y, vis_emb in loader:
            y = y.to(device)
            vis_emb = vis_emb.to(device)

            logits = model(vis_emb)
            pred = logits.argmax(dim=1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print(classification_report(all_labels, all_preds, digits=4))
    plot_confusion_matrix(cm,
                      class_names=list(class_map.keys()),
                      title="Visual-only Confusion Matrix")

# --------------------------------------------------------
# Fusion evaluation
# --------------------------------------------------------
def test_fusion(fusion_ckpt, root_dir, scenes, class_map, rgb_emb_path, num_classes, device):
    print("\n=== Fusion Evaluation ===")

    loader = get_fusion_test_loader(root_dir, scenes, class_map, rgb_emb_path)

    model = RadarVisionFusionNet(radar_ckpt_path="small_ra_cnn_best.pth",num_classes=num_classes).to(device)

    state = torch.load(fusion_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for ra, y, vis_emb in loader:
            ra = ra.to(device)
            y = y.to(device)
            vis_emb = vis_emb.to(device)

            logits = model(ra, vis_emb)
            pred = logits.argmax(dim=1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)
    print(classification_report(all_labels, all_preds, digits=4))
    plot_confusion_matrix(cm,
                      class_names=list(class_map.keys()),
                      title="Fusion Confusion Matrix")

# --------------------------------------------------------
# Main
# --------------------------------------------------------
if __name__ == "__main__":
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = ""
    rgb_emb_path = "minc_rgb_embeddings.npz"
    class_map = {'metal':0, 'glass':1, 'plastic':2, 'cardboard':3}
    num_classes = 4
    test_scenes = ['scene3']

    radar_ckpt = "small_ra_cnn_best.pth"
    visual_ckpt = "vision_only_best.pth"
    fusion_ckpt = "fusion_best.pth"

    test_radar_only(radar_ckpt, root_dir, test_scenes, class_map, num_classes, device)
    test_visual_only(visual_ckpt, root_dir, test_scenes, class_map, rgb_emb_path, num_classes, device)
    test_fusion(fusion_ckpt, root_dir, test_scenes, class_map, rgb_emb_path, num_classes, device)
