"""
Mini PointNet for per-point binary classification on LiDAR packets.

Input:  (B, 125, 3)  — azimuth, elevation, distance per point
Output: (B, 125)     — cone probability per point (sigmoid applied)
"""

import torch
import torch.nn as nn

class MiniPointNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        # per-point feature extraction: 3 -> 64 -> 128
        self.local_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # after concat with global (128 + 128 = 256): classify each point
        self.point_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),   # binary logit per point
        )

    def forward(self, x):
        """
        x: (B, N, 3)
        returns: (B, N) logits — apply sigmoid for probabilities
        """
        B, N, _ = x.shape

        # per-point features: (B, N, 128)
        x_flat = x.view(B * N, 3)
        local_feat = self.local_mlp(x_flat).view(B, N, 128)

        # global context: (B, 128)
        global_feat = local_feat.max(dim=1).values

        # broadcast global back to each point: (B, N, 256)
        global_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([local_feat, global_expanded], dim=-1)

        # per-point classification: (B, N, 1) -> (B, N)
        combined_flat = combined.view(B * N, 256)
        logits = self.point_mlp(combined_flat).view(B, N)

        return logits


class PointNetLoss(nn.Module):
    """BCE loss with positive class weighting for imbalanced cone/non-cone points."""
    def __init__(self, pos_weight=10.0):
        super().__init__()
        # pos_weight > 1 penalises missing cones more than false positives
        # tune this based on your observed positive rate
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight)
        )

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets.float())


if __name__ == "__main__":
    # quick sanity check
    model = MiniPointNet()
    x = torch.randn(8, 125, 3)
    logits = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")   # should be (8, 125)
    probs = torch.sigmoid(logits)
    print(f"Prob range: [{probs.min():.3f}, {probs.max():.3f}]")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")