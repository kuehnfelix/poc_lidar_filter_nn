import numpy as np
import torch
from neural_net.models import build_model
from neural_net.dataset import compute_features


class ConeDetector:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        checkpoint = torch.load(model_path, map_location=self.device)

        if "config" in checkpoint:
            self.config  = checkpoint["config"]
            self.model   = build_model(self.config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.features  = self.config["features"]
            self.threshold = checkpoint.get("threshold", 0.3)
            print(f"Loaded {self.config['model']} ({self.config['in_channels']}ch)  "
                  f"epoch={checkpoint.get('epoch','?')}  "
                  f"F1={checkpoint.get('f1','?'):.3f}  "
                  f"threshold={self.threshold}")
        else:
            raise ValueError("Checkpoint missing 'config' key — please retrain with the new train.py")

        self.model.eval()

    def predict(self, packet: np.ndarray) -> np.ndarray:
        """
        packet: (125, 3) raw az/el/dist
        returns: (125,) bool mask
        """
        return self.predict_batch(packet[np.newaxis])[0]

    def predict_proba(self, packet: np.ndarray) -> np.ndarray:
        """
        packet: (125, 3) raw az/el/dist
        returns: (125,) float probabilities
        """
        feat = compute_features(packet[np.newaxis], self.features)  # (1, 125, C)
        x = torch.from_numpy(feat).to(self.device)
        with torch.no_grad():
            return torch.sigmoid(self.model(x)).squeeze(0).cpu().numpy()

    def predict_batch(self, packets: np.ndarray) -> np.ndarray:
        """
        packets: (B, 125, 3) raw az/el/dist
        returns: (B, 125) bool mask
        """
        feat = compute_features(packets, self.features)  # (B, 125, C)
        x = torch.from_numpy(feat).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x))
        return (probs > self.threshold).cpu().numpy()
