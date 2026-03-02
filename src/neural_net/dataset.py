import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_features(packets: np.ndarray, features: list[str]) -> np.ndarray:
    """
    packets: (N, 125, 3) — raw az, el, dist
    returns: (N, 125, len(features))
    """
    az   = packets[:, :, 0]
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
    def __init__(self, dataset_path: str, features: list[str],
                 neg_keep_rate: float = 0.1):
        """
        neg_keep_rate: fraction of no-cone packets to keep (default 0.1 = 10%)
        """
        self.features = features
        self.samples  = []
        self.has_cone = []

        raw_data   = []
        raw_labels = []

        for path in sorted(glob.glob(f"{dataset_path}/**/*.npz", recursive=True)):
            f = np.load(path)
            raw_data.append(f["data"].astype(np.float32))
            raw_labels.append(f["labels"].astype(np.float32))

        all_data   = np.concatenate(raw_data,   axis=0)  # (M, 125, 3)
        all_labels = np.concatenate(raw_labels, axis=0)  # (M, 125)

        all_features = compute_features(all_data, features)  # (M, 125, C)

        has_cone_mask = all_labels.any(axis=1)  # (M,) bool

        pos_idx = np.where( has_cone_mask)[0]
        neg_idx = np.where(~has_cone_mask)[0]

        # keep all positives, subsample negatives
        n_neg_keep = max(1, int(len(neg_idx) * neg_keep_rate))
        neg_keep   = np.random.choice(neg_idx, size=n_neg_keep, replace=False)

        keep_idx = np.sort(np.concatenate([pos_idx, neg_keep]))

        for i in keep_idx:
            self.samples.append((all_features[i], all_labels[i]))
            self.has_cone.append(bool(has_cone_mask[i]))

        n_pos    = sum(self.has_cone)
        pos_rate = np.concatenate([all_labels[i] for i in keep_idx]).mean()

        print(f"Loaded {len(self.samples):,} packets "
              f"(kept all {len(pos_idx):,} positives + "
              f"{n_neg_keep:,}/{len(neg_idx):,} negatives "
              f"at {neg_keep_rate:.0%} keep rate)")
        print(f"Point-level positive rate: {pos_rate:.2%}  "
              f"→ recommended pos_weight: {(1-pos_rate)/pos_rate:.0f}")
        print(f"Features ({len(features)}ch): {features}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, labels = self.samples[idx]
        return torch.from_numpy(data), torch.from_numpy(labels)