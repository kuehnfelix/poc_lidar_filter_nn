import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_features(packets: np.ndarray, features: list[str]) -> np.ndarray:
    """
    packets: (N, 125, 3) — raw az, el, dist
    features: list of feature names to compute
    returns: (N, 125, len(features))
    """
    az   = packets[:, :, 0]   # (N, 125)
    el   = packets[:, :, 1]
    dist = packets[:, :, 2]

    feature_map = {
        "az":      az,
        "el":      el,
        "dist":    dist,
        "d_dist":  np.gradient(dist, axis=1),
        "dd_dist": np.gradient(np.gradient(dist, axis=1), axis=1),
        "d_az":    np.gradient(az,   axis=1),
        "d_el":    np.gradient(el,   axis=1),
    }

    channels = [feature_map[f] for f in features]
    return np.stack(channels, axis=-1).astype(np.float32)  # (N, 125, C)


class LiDARPacketDataset(Dataset):
    def __init__(self, dataset_path: str, features: list[str]):
        self.features  = features
        self.samples   = []
        self.has_cone  = []

        raw_data   = []
        raw_labels = []

        for path in sorted(glob.glob(f"{dataset_path}/**/*.npz", recursive=True)):
            f = np.load(path)
            raw_data.append(f["data"].astype(np.float32))      # (600, 125, 3)
            raw_labels.append(f["labels"].astype(np.float32))  # (600, 125)

        all_data   = np.concatenate(raw_data,   axis=0)  # (M, 125, 3)
        all_labels = np.concatenate(raw_labels, axis=0)  # (M, 125)

        # compute all features in one vectorised pass
        all_features = compute_features(all_data, features)  # (M, 125, C)

        for i in range(len(all_features)):
            self.samples.append((all_features[i], all_labels[i]))
            self.has_cone.append(bool(all_labels[i].any()))

        n_pos    = sum(self.has_cone)
        pos_rate = all_labels.mean()
        print(f"Loaded {len(self.samples):,} packets — "
              f"{n_pos:,} with cones ({n_pos/len(self.samples):.1%})")
        print(f"Point-level positive rate: {pos_rate:.2%}  "
              f"→ recommended pos_weight: {(1-pos_rate)/pos_rate:.0f}")
        print(f"Features ({len(features)}ch): {features}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, labels = self.samples[idx]
        return torch.from_numpy(data), torch.from_numpy(labels)
