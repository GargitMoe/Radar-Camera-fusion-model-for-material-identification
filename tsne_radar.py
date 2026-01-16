# tsne_radar.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from model import SmallRACNN
from dataset import RadarRADataset
from config import CLASS_MAP, TRAIN_SCENES, VAL_SCENES

ROOT_DIR = ""  # 雷达 .mat 的路径
MODEL_PATH = "small_ra_cnn_best.pth"
BATCH_SIZE = 16

def extract_embeddings():
    dataset = RadarRADataset(
        root_dir=ROOT_DIR,
        scene_list=TRAIN_SCENES + VAL_SCENES,
        class_map=CLASS_MAP,
        transform=None
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SmallRACNN(num_classes=len(CLASS_MAP))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    all_emb = []
    all_labels = []

    with torch.no_grad():
        for ra, label in loader:
            feat32 = model.extract_feat(ra)  # [B,32]
            all_emb.append(feat32.numpy())
            all_labels.append(label.numpy())

    all_emb = np.concatenate(all_emb, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_emb, all_labels


def run_tsne():
    emb, labels = extract_embeddings()
    print("Embedding shape:", emb.shape)

    tsne = TSNE(
        n_components=2,
        learning_rate='auto',
        init='pca',
        perplexity=10,
        random_state=42
    )
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(8, 8))
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=f"class {c}", alpha=0.7)

    plt.legend()
    plt.title("Radar Embedding t-SNE (32D → 2D)")
    plt.show()


if __name__ == "__main__":
    run_tsne()
