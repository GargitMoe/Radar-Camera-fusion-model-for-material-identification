# dataset_fusion.py

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from dataset import RadarRADataset  

class RadarFusionDataset(Dataset):
    def __init__(self,
                 root_dir,
                 scene_list=None,
                 class_map=None,
                 rgb_emb_path: str = None,
                 transform=None):
        self.base_dataset = RadarRADataset(
            root_dir=root_dir,
            scene_list=scene_list,
            class_map=class_map,
            transform=transform
        )

        if rgb_emb_path is None:
            raise ValueError("rgb_emb_path not found")

        rgb_npz = np.load(rgb_emb_path)
        self.rgb_emb_dict = {k: rgb_npz[k] for k in rgb_npz.files}

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        ra_tensor, label = self.base_dataset[idx]

        mat_path, cls_name, scene = self.base_dataset.samples[idx]
        stem = Path(mat_path).stem 
        parts = stem.split("_")      # ["adc","data","1021","67","1","25","0","ch01"]
        date  = parts[2]             # "1021"
        mid   = parts[3].zfill(3)    # "067"
        img_id = date + mid          # "1021067"
        key = f"{scene}/{img_id}"


        vis_emb_np = self.rgb_emb_dict[key]           # (768,)
        vis_emb = torch.from_numpy(vis_emb_np).float()  # [768]

        return ra_tensor, label, vis_emb
