import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch

class RadarRADataset(Dataset):
    def __init__(self, root_dir, scene_list=None, class_map=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if scene_list is None:
            scene_list = ['scene1', 'scene2', 'scene3']
        if class_map is None:
            class_map = {
                'metal': 0,
                'glass': 1,
                'plastic': 2,
                'cardboard': 3,
            }
        self.class_map = class_map

        self.samples = []
        for scene in scene_list:
            scene_dir = os.path.join(root_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            for cls_name in os.listdir(scene_dir):
                cls_dir = os.path.join(scene_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                if cls_name not in class_map:
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.endswith('.mat'):
                        path = os.path.join(cls_dir, fname)
                        self.samples.append((path, cls_name, scene))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mat_path, cls_name, scene = self.samples[idx]
        mat = loadmat(mat_path)

        ra = mat['ra'].astype(np.float32)   # H × W, [0,1]
        label = self.class_map[cls_name]

        # to tensor, add channel dim: 1×H×W
        ra_tensor = torch.from_numpy(ra)[None, :, :]

        if self.transform is not None:
            ra_tensor = self.transform(ra_tensor)

        return ra_tensor, label

def ra_augment(x: torch.Tensor) -> torch.Tensor:
    # 确保是 float
    x = x.float()

    # 1) 随机整体增益（乘一个接近1的系数）
    if torch.rand(1).item() < 0.7:
        gain = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # [0.9, 1.1]
        x = x * gain

    # 2) 随机整体偏移（加一个小的偏置）
    if torch.rand(1).item() < 0.7:
        offset = (torch.rand(1).item() - 0.5) * 0.1      # [-0.05, 0.05]
        x = x + offset

    # 3) 加一点高斯噪声
    if torch.rand(1).item() < 0.7:
        noise_std = 0.02
        noise = torch.randn_like(x) * noise_std
        x = x + noise

    # 4) 在角度/距离方向做小范围roll（平移），不改变尺寸
    if torch.rand(1).item() < 0.5:
        # 角度方向 roll 1~2 个像素
        shift_h = int(torch.randint(-2, 3, (1,)).item())
        x = torch.roll(x, shifts=shift_h, dims=-2)
    if torch.rand(1).item() < 0.5:
        # 距离方向 roll 1~2 个像素
        shift_w = int(torch.randint(-2, 3, (1,)).item())
        x = torch.roll(x, shifts=shift_w, dims=-1)

    # 5) 截断回 [0,1]
    x = x.clamp(0.0, 1.0)

    return x
