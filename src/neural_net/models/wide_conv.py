import torch
import torch.nn as nn


class WideConvNet(nn.Module):
    """
    1D conv per-point binary classifier with growing receptive field.

    Input:  (B, in_channels, 125) — channels first
            also accepts (B, 125, in_channels) — auto-permutes
    Output: (B, 125) — raw logits (use BCEWithLogitsLoss for training)

    Receptive field per layer: 7 -> 11 -> 15 -> 19 -> 25 -> 25
    """
    def __init__(self, in_channels=5):
        super().__init__()
        self.in_channels = in_channels

        self.net = nn.Sequential(
            # Conv1: k=7, 3->32, ReLU,       RF=7
            nn.Conv1d(in_channels, 32,  kernel_size=7,  padding=3),
            nn.ReLU(inplace=True),

            # Conv2: k=5, 32->64, ReLU,      RF=11
            nn.Conv1d(32,  64,  kernel_size=5,  padding=2),
            nn.ReLU(inplace=True),

            # Conv3: k=5, 64->64, ReLU,      RF=15
            nn.Conv1d(64,  64,  kernel_size=5,  padding=2),
            nn.ReLU(inplace=True),

            # Conv4: k=5, 64->128, ReLU+BN,  RF=19
            nn.Conv1d(64,  128, kernel_size=5,  padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Conv5: k=7, 128->128, ReLU+BN, RF=25
            nn.Conv1d(128, 128, kernel_size=7,  padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # Conv6: k=1, 128->1, logit output
            nn.Conv1d(128, 1,   kernel_size=1),
        )

    def forward(self, x):
        # auto-permute if channels-last (B, 125, C)
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 2, 1)
        return self.net(x).squeeze(1)  # (B, 125) logits


if __name__ == "__main__":
    model = WideConvNet(in_channels=5)
    x = torch.randn(8, 125, 5)
    logits = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
