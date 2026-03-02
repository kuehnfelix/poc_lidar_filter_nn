# Neural Network

## Features

| Feature | Description |
|---|---|
| `az` | Azimuth (rad) |
| `el` | Elevation (rad) |
| `dist` | Distance (m) |
| `d_dist` | ∂dist/∂point — distance gradient along scan |
| `dd_dist` | ∂²dist — curvature, distinctive for cone shape |

## Models

| Model | Params | Input | Notes |
|---|---|---|---|
| `DilatedConvNet` | ~30k | `(B, C, 125)` | Fastest inference |
| `WideConvNet` | ~150k | `(B, C, 125)` | RF=25, BN on deep layers |
| `PointNetPP` | ~300k | `(B, 125, C)` | Global context via max+mean pool |

All output raw logits. Use `BCEWithLogitsLoss`.

## Training
```bash
python -m neural_net.train neural_net.configs.wide_conv_5ch --dataset dataset
python -m neural_net.train neural_net.configs.wide_conv_5ch --dataset dataset --resume checkpoints/wide_conv_5ch.pt
```

## Class Imbalance (~2% positive points)

| Setting | Value | Purpose |
|---|---|---|
| `pos_weight` | 42 | Penalise missed cones |
| `oversample_factor` | 10 | See cone packets more often |
| `threshold` | 0.3 | Lower than 0.5 for better recall |
| `neg_keep_rate` | 0.1 | Discard 90% of all-negative packets |

Save criterion is best **F1**, not best loss.

## Inference
```python
from neural_net.inference import ConeDetector
detector = ConeDetector("checkpoints/wide_conv_5ch.pt")
mask  = detector.predict(packet)        # (125,3) → (125,) bool
probs = detector.predict_proba(packet)  # (125,) float
masks = detector.predict_batch(packets) # (B,125,3) → (B,125) bool
```

Config, features and threshold are loaded from checkpoint automatically.

## Jetson Orin AGX (estimated)

| Model | Single packet | 600 batched | TensorRT FP16 |
|---|---|---|---|
| DilatedConvNet | ~0.05ms | ~2ms | ~1ms |
| WideConvNet | ~0.08ms | ~3ms | ~1.5ms |
| PointNetPP | ~0.3ms | ~8ms | ~4ms |

At 6000 Hz, use single-packet inference with adaptive batching queue.