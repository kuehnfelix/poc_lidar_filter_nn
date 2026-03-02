import torch
import torch.nn as nn


class DilatedConvNet(nn.Module):
    """
    1D dilated conv per-point binary classifier.

    Input:  (B, in_channels, 125)  — channels first
    Output: (B, 125)               — raw logits

    Accepts (B, 125, in_channels) too — auto-permutes if needed.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, dilation=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=5, dilation=3, padding=6),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        # auto-permute if channels-last (B, 125, C)
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 2, 1)
        return self.net(x).squeeze(1)  # (B, 125) logits
