import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=42.0):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets.float())
