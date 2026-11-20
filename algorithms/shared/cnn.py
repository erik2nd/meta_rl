"""
Compact CNN encoder для Procgen 64x64x3 -> feature vector.

На CPU полноценный IMPALA-CNN слишком тяжёл — используем упрощённую версию
(NatureCNN из DQN paper), которой хватает для coinrun-easy.
"""
import torch
import torch.nn as nn


class NatureCNN(nn.Module):
    """
    Стандартный CNN backbone из DQN. На coinrun-easy работает.
    Input:  (B, 64, 64, 3) uint8 или (B, 3, 64, 64) float
    Output: (B, out_dim) float
    """
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),  # 64->15
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), # 15->6
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), # 6->4
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            flat_dim = self.conv(dummy).flatten(1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        # Приводим к (B, 3, 64, 64) float
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.shape[-1] == 3:  # NHWC -> NCHW
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)
