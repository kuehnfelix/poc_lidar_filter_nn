# Dataset Generation

## Format

Each `.npz` file contains one frame:
- `data`: `(600, 125, 3)` float32 — az_rad, el_rad, dist per point
- `labels`: `(600, 125)` uint8 — 1 = cone point, 0 = other

## Synthetic Generation
```bash
python dataset_generator.py
```

Generates random scenes with cones along a track. Each frame uses a random lidar pose with noise (lateral, angular, height errors).
