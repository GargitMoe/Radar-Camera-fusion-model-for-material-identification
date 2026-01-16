# vision_head.py
import torch.nn as nn
import torch.nn.functional as F

class VisionHeadLinear(nn.Module):
    def __init__(self, vis_dim=768, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(vis_dim, num_classes)

    def forward(self, x):
        # x: [B, vis_dim]
        return self.fc(x)

class VisionHeadMLP(nn.Module):
    def __init__(self, vis_dim=768, hidden_dim=128, num_classes=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(vis_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, return_feat=False):
        h = F.relu(self.fc1(x))      # [B, hidden_dim]
        h = self.dropout(h)
        logits = self.fc2(h)         # [B, num_classes]
        if return_feat:
            return logits, h
        return logits
