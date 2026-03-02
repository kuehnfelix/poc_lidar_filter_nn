"""
Per-point binary classifiers for LiDAR cone detection.

Two models available:
  - PointNetPP:      (B, 125, 3)  -> (B, 125) logits — larger, global context
  - DilatedConvNet:  (B, 3, 125)  -> (B, 125) logits — smaller, faster, channels-first
"""

import torch
import torch.nn as nn


# ── PointNet++ ─────────────────────────────────────────────────────────────────

class SharedMLP(nn.Module):
    """Linear -> BN -> ReLU block operating on (B*N, C)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PointNetPP(nn.Module):
    """
    Input:  (B, 125, 3)  — azimuth, elevation, distance per point (channels last)
    Output: (B, 125)     — raw logits per point

    Architecture:
      - Three-stage local MLP with skip connection: 3 -> 64 -> 128 -> 256
      - Dual global aggregation (max + mean): 256 + 256 = 512
      - Per-point classifier with global context: 768 -> 256 -> 128 -> 64 -> 1
      ~300k parameters
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.mlp1 = SharedMLP(3,   64)
        self.mlp2 = SharedMLP(64,  128)
        self.mlp3 = SharedMLP(128, 256)
        self.skip = nn.Linear(3, 256)
        self.cls1 = SharedMLP(768, 256)
        self.cls2 = SharedMLP(256, 128)
        self.cls3 = SharedMLP(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        B, N, _ = x.shape
        x_flat = x.reshape(B * N, 3)

        local_feat = self.mlp3(self.mlp2(self.mlp1(x_flat))) + self.skip(x_flat)
        local_3d   = local_feat.view(B, N, 256)

        global_max  = local_3d.max(dim=1).values.unsqueeze(1).expand(-1, N, -1)
        global_mean = local_3d.mean(dim=1).unsqueeze(1).expand(-1, N, -1)

        combined = torch.cat([local_3d, global_max, global_mean], dim=-1)
        c = self.cls3(self.cls2(self.cls1(combined.reshape(B * N, 768))))
        return self.out(self.dropout(c)).view(B, N)


# ── Dilated Conv Net ───────────────────────────────────────────────────────────

class DilatedConvNet(nn.Module):
    """
    Input:  (B, C, 125)  — channels first (transpose of PointNet input)
    Output: (B, 125)     — raw logits per point

    Architecture:
      Conv1D(C->32, k=5, d=1) -> ReLU
      Conv1D(32->64, k=5, d=2) -> ReLU
      Conv1D(64->64, k=5, d=3) -> ReLU
      Conv1D(64->1, k=1)
      ~30k parameters — much faster at inference than PointNet
    """
    def __init__(self, in_channels=3):
        super().__init__()
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
        x = x.permute(0, 2, 1)
        return self.net(x).squeeze(1)  # (B, 125) logits


# ── Shared loss ────────────────────────────────────────────────────────────────

class PointNetLoss(nn.Module):
    """BCE loss with positive class weighting for ~1% cone points."""
    def __init__(self, pos_weight=99.0):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets.float())


MiniPointNet = PointNetPP  # backwards compat alias


if __name__ == "__main__":
    for name, model, x in [
        ("PointNetPP",     PointNetPP(),          torch.randn(8, 125, 3)),
        ("DilatedConvNet", DilatedConvNet(),       torch.randn(8, 3, 125)),
    ]:
        logits = model(x)
        n = sum(p.numel() for p in model.parameters())
        print(f"{name:20s}  input={tuple(x.shape)}  output={tuple(logits.shape)}  params={n:,}")