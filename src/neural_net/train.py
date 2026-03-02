"""
Usage:
    python train.py configs.dilated_5ch
    python train.py configs.dilated_5ch --resume checkpoints/dilated_5ch.pt
"""

import os
import sys
import importlib
import argparse
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from neural_net.dataset import LiDARPacketDataset
from neural_net.models import build_model, WeightedBCELoss


def train(config, dataset_path="dataset", resume=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    full_dataset = LiDARPacketDataset(dataset_path, config["features"])
    n_val   = int(len(full_dataset) * config["val_split"])
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_weights = [
        config["oversample_factor"] if full_dataset.has_cone[i] else 1.0
        for i in train_ds.indices
    ]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False,    num_workers=4, pin_memory=True)

    # model
    model     = build_model(config).to(device)
    criterion = WeightedBCELoss(pos_weight=config["pos_weight"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    start_epoch = 1
    best_f1     = 0.0
    threshold   = config["threshold"]

    # resume
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_f1     = checkpoint["f1"]
            threshold   = checkpoint.get("threshold", threshold)
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed from {resume} — continuing from epoch {start_epoch}")

    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)

    for epoch in range(start_epoch, start_epoch + config["epochs"]):
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
                "config":          config,
            }, config["save_path"])
            print(f"  ✓ saved (F1={f1:.3f})")

    print(f"\nDone. Best F1: {best_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config",           help="config module, e.g. neural_net.configs.dilated_5ch")
    parser.add_argument("--dataset",        default="dataset")
    parser.add_argument("--resume",         default=None)
    args = parser.parse_args()

    cfg = importlib.import_module(args.config).config
    train(cfg, dataset_path=args.dataset, resume=args.resume)
