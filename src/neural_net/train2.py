"""
Training script for PointNetPP cone classifier.
Handles ~1% positive rate with pos_weight + oversampling.
Supports resuming from a checkpoint with resume="model.pt".

Dataset layout expected:
    dataset/
        track_00/
            frame_000.npz   # data: (600, 125, 3), labels: (600, 125)
            frame_001.npz
            ...
        track_01/
            ...
"""
from pathlib import Path
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from neural_net.model2 import MiniPointNet, PointNetLoss, DilatedConvNet


# ── Dataset ────────────────────────────────────────────────────────────────────

class LiDARPacketDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.samples  = []
        self.has_cone = []

        for path in sorted(glob.glob(f"{dataset_path}/**/*.npz", recursive=True)):
            f = np.load(path)
            data   = f["data"].astype(np.float32)    # (600, 125, 3)
            labels = f["labels"].astype(np.float32)  # (600, 125)
            for data_packet, label_packet in zip(data, labels):
                if np.sum(label_packet) == 0 and np.random.rand() > 0.1:
                    continue
                self.samples.append((data_packet, label_packet))
                self.has_cone.append(bool(label_packet.any()))

        n_pos = sum(self.has_cone)
        print(f"Loaded {len(self.samples):,} packets — "
              f"{n_pos:,} with cones ({n_pos/len(self.samples):.1%}), "
              f"{len(self.samples)-n_pos:,} without")

        all_labels = np.concatenate([s[1] for s in self.samples])
        pos_rate = all_labels.mean()
        print(f"Point-level positive rate: {pos_rate:.2%}  "
              f"→ recommended pos_weight: {(1-pos_rate)/pos_rate:.0f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, labels = self.samples[idx]
        return torch.from_numpy(data), torch.from_numpy(labels)


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    dataset_path      = "dataset",
    epochs            = 40,
    batch_size        = 64,
    lr                = 1e-3,
    pos_weight        = 42.0,
    oversample_factor = 10,
    threshold         = 0.3,
    val_split         = 0.1,
    save_path         = "model_conv.pt",
    resume            = None,    # set to "model.pt" to continue from checkpoint
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    full_dataset = LiDARPacketDataset(dataset_path)
    n_val   = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_weights = [
        oversample_factor if full_dataset.has_cone[i] else 1.0
        for i in train_ds.indices
    ]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=4, pin_memory=True)

    # model, optimizer, scheduler
    model     = DilatedConvNet().to(device)
    criterion = PointNetLoss(pos_weight=pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    start_epoch = 1
    best_f1     = 0.0

    # ── resume from checkpoint ──────────────────────────────────────────────
    if resume is not None and Path(resume).is_file():
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1     = checkpoint["f1"]
        threshold   = checkpoint.get("threshold", threshold)
        print(f"Resumed from {resume}  "
              f"(epoch {checkpoint['epoch']}, F1={best_f1:.3f}) "
              f"— continuing from epoch {start_epoch}")
    # ────────────────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, start_epoch + epochs):
        # train
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validate
        model.eval()
        val_loss = 0.0
        tp = fp = fn = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                logits = model(data)
                val_loss += criterion(logits, labels).item()
                preds = (torch.sigmoid(logits) > threshold).long()
                lbls  = labels.long()
                tp += ((preds == 1) & (lbls == 1)).sum().item()
                fp += ((preds == 1) & (lbls == 0)).sum().item()
                fn += ((preds == 0) & (lbls == 1)).sum().item()

        val_loss  /= len(val_loader)
        precision  = tp / (tp + fp + 1e-8)
        recall     = tp / (tp + fn + 1e-8)
        f1         = 2 * precision * recall / (precision + recall + 1e-8)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  "
              f"(TP={tp} FP={fp} FN={fn})")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch":           epoch,
                "f1":              f1,
                "threshold":       threshold,
            }, save_path)
            print(f"  ✓ saved (F1={f1:.3f})")

    print(f"\nDone. Best F1: {best_f1:.3f}")


if __name__ == "__main__":
    train(resume="model_conv.pt", lr=1e-5, epochs=500)