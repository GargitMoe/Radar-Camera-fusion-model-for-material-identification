# tsne_visual_head.py

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from dataset_fusion import RadarFusionDataset
from vision_head import VisionHeadMLP

CLASS_MAP = {
    'metal': 0,
    'glass': 1,
    'plastic': 2,
    'cardboard': 3,
}

TRAIN_SCENES = ['scene1', 'scene2']
VAL_SCENES   = ['scene3']

ROOT_DIR     = ""
RGB_EMB_PATH = "minc_rgb_embeddings.npz"
BATCH_SIZE   = 32

BEST_PATH = "vision_only_best.pth"

def build_val_loader():
    ds = RadarFusionDataset(
        root_dir=ROOT_DIR,
        scene_list=VAL_SCENES,   # 只看 scene3 的分布
        class_map=CLASS_MAP,
        rgb_emb_path=RGB_EMB_PATH,
        transform=None,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return ds, loader

def collect_head_features():
    ds, loader = build_val_loader()
    model = VisionHeadMLP(vis_dim=768, hidden_dim=128, num_classes=len(CLASS_MAP))
    state = torch.load(BEST_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    all_feat = []
    all_labels = []

    with torch.no_grad():
        for ra, y, vis_emb in loader:
            logits, feat128 = model(vis_emb, return_feat=True)
            all_feat.append(feat128.numpy())
            all_labels.append(y.numpy())

    all_feat   = np.concatenate(all_feat, axis=0)      # [N,128]
    all_labels = np.concatenate(all_labels, axis=0)    # [N]
    return all_feat, all_labels

def run_tsne():
    X, y = collect_head_features()
    print("Head feature shape:", X.shape)

    tsne = TSNE(
        n_components=2,
        learning_rate='auto',
        init='pca',
        perplexity=5,
        random_state=42,
    )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 8))
    class_names = {0: 'metal', 1: 'glass', 2: 'plastic', 3: 'cardboard'}
    for c in np.unique(y):
        idx = (y == c)
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=class_names[int(c)], alpha=0.7)

    plt.legend()
    plt.title("t-SNE of visual head hidden features (128-d)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_tsne()
