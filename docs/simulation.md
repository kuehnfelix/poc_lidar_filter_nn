# Simulation

## Scene Objects

All objects inherit `SceneObject` and implement `intersect_batch(origins, directions)` taking `(N,3)` NumPy arrays:

| Class | Description |
|---|---|
| `ConeObject` | Traffic cone |
| `CylinderObject` | Cylinder with caps |
| `BoxObject` | Oriented bounding box |
| `GroundPlane` | Infinite horizontal plane |

## Scene
```python
scene = Scene()
scene.add_object(ConeObject(...))
hits = scene.intersect_many_arrays(origins, directions)  # (N,3) → N Hits
```

Uses `ThreadPoolExecutor` — each object tested in parallel (NumPy releases GIL).

## LiDAR Model
```python
lidar = LiDARModel(scan_pattern=ZigZagScanPattern())
hits, rays = lidar.measure_frame(scene, lidar_pose)  # fires all 75,000 rays at once
```

## Scan Patterns

**`ZigZagScanPattern`** — synthetic, matches real sensor FOV (±62° az, ±12° el)

**`RealRecordedScanPattern`** — uses real recorded az/el, random frame per iteration
```python
pattern = RealRecordedScanPattern("dataset/record2")
lidar = LiDARModel(scan_pattern=pattern, zero_channels=True)
```
> Set `zero_channels=True` — beam offsets are already in the recorded data.