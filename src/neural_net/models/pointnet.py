import torch
import torch.nn as nn


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
    Input:  (B, 125, C) — channels last
    Output: (B, 125)    — raw logits

    Architecture:
      - Three-stage local MLP with skip connection: C -> 64 -> 128 -> 256
      - Dual global aggregation (max + mean): 256 + 256 = 512
      - Per-point classifier with global context: 768 -> 256 -> 128 -> 64 -> 1
      ~300k parameters
    """
    def __init__(self, in_channels=3, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels

        self.mlp1 = SharedMLP(in_channels, 64)
        self.mlp2 = SharedMLP(64,  128)
        self.mlp3 = SharedMLP(128, 256)
        self.skip = nn.Linear(in_channels, 256)

        self.cls1 = SharedMLP(768, 256)
        self.cls2 = SharedMLP(256, 128)
        self.cls3 = SharedMLP(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        B, N, _ = x.shape
        x_flat = x.reshape(B * N, self.in_channels)

        local_feat = self.mlp3(self.mlp2(self.mlp1(x_flat))) + self.skip(x_flat)
        local_3d   = local_feat.view(B, N, 256)

        global_max  = local_3d.max(dim=1).values.unsqueeze(1).expand(-1, N, -1)
        global_mean = local_3d.mean(dim=1).unsqueeze(1).expand(-1, N, -1)

        combined = torch.cat([local_3d, global_max, global_mean], dim=-1)
        c = self.cls3(self.cls2(self.cls1(combined.reshape(B * N, 768))))
        return self.out(self.dropout(c)).view(B, N)


if __name__ == "__main__":
    model = PointNetPP(in_channels=5)
    x = torch.randn(8, 125, 5)
    logits = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
