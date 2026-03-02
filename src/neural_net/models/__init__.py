from .dilated_conv import DilatedConvNet
from .wide_conv import WideConvNet
from .pointnet import PointNetPP
from .loss import WeightedBCELoss

MODELS = {
    "dilated_conv": DilatedConvNet,
    "wide_conv":    WideConvNet,
    "pointnet":     PointNetPP,
}

def build_model(config):
    cls = MODELS[config["model"]]
    return cls(in_channels=config["in_channels"])
