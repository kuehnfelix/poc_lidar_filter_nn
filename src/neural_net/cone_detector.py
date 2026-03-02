"""
Inference wrapper for the trained cone classifier.

Usage:
    from inference import ConeDetector

    detector = ConeDetector("model.pt")
    
    # from a single packet (125 points)
    mask = detector.predict(packet_data)          # (125,) bool array

    # from a Hit/Ray list (as returned by lidar_model.measure_frame)
    mask = detector.predict_from_hits(hits, rays) # (N,) bool array
"""

import numpy as np
import torch
from neural_net.model2 import MiniPointNet, DilatedConvNet


class ConeDetector:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = DilatedConvNet().to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        # use the threshold that was optimal during training
        self.threshold = checkpoint.get("threshold", 0.3)
        print(f"Loaded model from {model_path}  "
              f"(epoch {checkpoint.get('epoch','?')}, "
              f"F1={checkpoint.get('f1', '?'):.3f}, "
              f"threshold={self.threshold})")

    def predict(self, packet: np.ndarray) -> np.ndarray:
        """
        packet: (125, 3) float array — azimuth, elevation, distance
        returns: (125,) bool array — True where a cone point is detected
        """
        x = torch.from_numpy(packet.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, 125, 3)
        with torch.no_grad():
            logits = self.model(x)                          # (1, 125)
            probs  = torch.sigmoid(logits).squeeze(0)       # (125,)
        return (probs > self.threshold).cpu().numpy()

    def predict_proba(self, packet: np.ndarray) -> np.ndarray:
        """Same as predict() but returns raw probabilities instead of a bool mask."""
        x = torch.from_numpy(packet.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.sigmoid(logits).squeeze(0)
        return probs.cpu().numpy()

    def predict_batch(self, packets: np.ndarray) -> np.ndarray:
        """
        packets: (B, 125, 3) array of multiple packets
        returns: (B, 125) bool array
        """
        x = torch.from_numpy(packets.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.sigmoid(logits)
        return (probs > self.threshold).cpu().numpy()

    def predict_from_hits(self, hits, rays) -> np.ndarray:
        """
        Convenience method: takes raw hits/rays from lidar_model.measure_frame
        and returns a bool mask of the same length.

        hits: flat list of Hit objects (length N)
        rays: flat list of Ray objects (length N)
        returns: (N,) bool array
        """
        N = len(hits)
        # build packet array — zero for missed rays
        packet = np.zeros((N, 3), dtype=np.float32)
        for i, (hit, ray) in enumerate(zip(hits, rays)):
            if hit.hit:
                packet[i] = (ray.azimuth, ray.elevation, hit.distance)

        # pad or trim to 125 points if needed
        if N != 125:
            raise ValueError(f"Expected 125 points per packet, got {N}. "
                             "Use predict() or predict_batch() directly for other sizes.")

        return self.predict(packet)