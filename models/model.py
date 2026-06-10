# import torch.nn as nn
# import torch.nn.functional as F

# class SmallRACNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm2d(8)

#         self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
#         self.bn2   = nn.BatchNorm2d(16)

#         self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.bn3   = nn.BatchNorm2d(32)

#         self.pool  = nn.MaxPool2d(2, 2)
#         self.gap   = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc    = nn.Linear(32, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.gap(x)               # [B,32,1,1]
#         x = x.view(x.size(0), -1)     # [B,32]
#         x = self.fc(x)
#         return x
import torch.nn as nn
import torch.nn.functional as F

class SmallRACNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)

        self.pool  = nn.MaxPool2d(2, 2)
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(32, num_classes)

    def extract_feat(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)               # [B,32,1,1]
        x = x.view(x.size(0), -1)     # [B,32]
        return x

    def forward(self, x):
        feat = self.extract_feat(x)   # [B,32]
        logits = self.fc(feat)        # [B,num_classes]
        return logits

