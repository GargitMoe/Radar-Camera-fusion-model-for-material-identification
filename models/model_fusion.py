import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SmallRACNN  

class RadarVisionFusionNet(nn.Module):
    def __init__(self,
                 radar_ckpt_path: str,
                 num_classes: int = 4,
                 radar_feat_dim: int = 32,
                 vis_dim: int = 768,
                 hidden_dim: int = 128,
                 freeze_radar: bool = True):
        super().__init__()

        self.radar_encoder = SmallRACNN(num_classes=num_classes)
        state = torch.load(radar_ckpt_path, map_location="cpu")
        self.radar_encoder.load_state_dict(state)

        if freeze_radar:
            for p in self.radar_encoder.parameters():
                p.requires_grad = False

        self.radar_proj = nn.Sequential(
            nn.Linear(radar_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.ReLU(),
        )

        # fusion head
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, ra_tensor, vis_emb):

        # calculate the embedding
        radar_feat32 = self.radar_encoder.extract_feat(ra_tensor)   # [B,32]

        # projection
        radar_z = self.radar_proj(radar_feat32)  # [B,hidden_dim]
        vis_z   = self.vis_proj(vis_emb)         # [B,hidden_dim]

        fused = torch.cat([radar_z, vis_z], dim=-1)  # [B,2*hidden_dim]
        logits = self.classifier(fused)              # [B,num_classes]

        return logits
