LiDAR Cone Point Classification — Proof of Concept
=================================================

Overview
--------
This project implements a proof-of-concept pipeline for training a lightweight
neural network to classify LiDAR points that lie on traffic cones. The system
generates synthetic LiDAR packets using a simple CPU-based raycaster,
automatically labels cone vs non-cone points, and (eventually) trains a small
1-D CNN that operates directly on incoming LiDAR packets.

Design choices favor fast iteration, deterministic labeling and minimal
dependencies so the codebase remains easy to extend with higher-fidelity
backends later (BVH, GPU raytracing, Unreal, Chrono, etc.).

Problem statement
-----------------
- Input: LiDAR packet of 125 ordered points (5 channels × 25 points), delivered
	as a nearly-colinear scanline rather than a 2-D image.
- Goal: classify each point as cone / not-cone in real time.
- Approach: train a small network that processes each 125-point packet
	independently and outputs per-point binary labels.

Project layout (approx)
-----------------------
- `data/` — generated datasets (expected: `train.pt`, `val.pt`, `test.pt`).
- `src/` — core source tree
	- `lidar_sim/scene` — scene container, objects and procedural generation
	- `lidar_sim/lidar` — LiDAR model + simulator (scan → features + labels)
	- `lidar_sim/dataset` — dataset generator that calls the scene + lidar
	- `neural_net/` — network definition and training loop (1-D CNN)
	- `scripts/` — convenience entrypoints for dataset generation / training

Scene generation
----------------
Each generated scene contains a ground plane, a procedural track layout and
traffic cones placed along the track. Random clutter (boxes, cylinders, fences)
and an articulated vehicle pose sampled along the track are also supported so
that each scene yields varied scans and labels.

LiDAR model and raycasting
--------------------------
- A fixed scan pattern (azimuth/elevation pairs) defines the 125 rays per
	packet. Each ray is cast from the vehicle pose into the scene and the nearest
	analytic intersection is returned.
- Current analytic backends include ray/plane (ground), ray/cone, ray/cylinder
	and ray/box. The implementation is deliberately brute-force for this POC.

Automatic labeling
------------------
Labeling is deterministic: a return is labeled `1` if the nearest intersected
object is a traffic cone, otherwise `0`. No heuristics or post-processing are used.

Dataset format
--------------
- Stored in ML-friendly formats (e.g. `.pt`, `.npz`).
- Features: `X` shaped `(N, 125, F)` where F are per-point features such as
	distance, azimuth, elevation, intensity, channel id.
- Labels: `y` shaped `(N, 125)`, binary per-point.

Neural network
--------------
- A compact 1-D convolutional network operating on the ordered 125-point
	scanline. Convolutions run along point order and produce per-point outputs.
	This matches the physical scan pattern and keeps latency low for embedded use.

Current implementation status
-----------------------------
Implemented:
- Geometry primitives and analytic intersection code for cones and ground.
- `Hit` and `ObjectType` core types, `Scene.intersect()` (brute-force nearest-hit).
- `lidar_simulator.scan()` which consumes rays and builds features/labels.
- A dataset generator skeleton that stacks per-scene packets into arrays.

Partially implemented / TODO (blocks a runnable end-to-end flow):
- `src/lidar_sim/core/ray.py` — missing the `Ray` type (currently empty).
- `src/lidar_sim/lidar/lidar_model.py` — `LiDARModel.generate_rays()` is TODO
	(ray generation + scan pattern should be implemented here).
- `src/lidar_sim/scene/scene_generator.py` — scene population methods are
	placeholders and must be filled to produce varied scenes automatically.
- `neural_net/` (`model.py`, `train.py`) — currently empty; needs network and
	training loop implementations.
- `scripts/` and `configs/` — currently empty convenience scripts and YAML
	configs. Additions here will provide end-to-end CLI entrypoints.

Status summary
--------------
This repository is a well-structured proof-of-concept with working geometry
and intersection code, but not yet runnable end-to-end. The missing pieces are
small and localized: add a `Ray` type, implement `LiDARModel.generate_rays()`,
fill the scene generator, and add a minimal neural-net/training script to make
the pipeline produce and consume datasets.

Next steps
--------------------------------------
1. Implement `src/lidar_sim/core/ray.py` (simple `Ray(origin, direction)`).
2. Implement `LiDARModel.generate_rays()` in
	 `src/lidar_sim/lidar/lidar_model.py` using a fixed list of (az, el) angles
	 and returning `Ray` objects in world coordinates.
3. Implement scene population in
	 `src/lidar_sim/scene/scene_generator.py` (ground + a few cones + pose).
4. Add a minimal `scripts/generate_dataset.py` that uses
	 `lidar_sim.dataset.dataset_generator.DatasetGenerator.generate()` and saves
	 results to `data/*.npz` or `data/*.pt`.
5. Add a simple `neural_net/model.py` and `neural_net/train.py` (small 1-D CNN)
	 and a `scripts/train_model.py` entrypoint.

Quick local commands (example)
------------------------------
After implementing the small TODOs above, typical commands might be:

```bash
python scripts/generate_dataset.py --out data/train.npz --num 1000
python scripts/train_model.py --data data/train.npz --epochs 20
```

Notes & next steps
------------------
- Add configurable noise models (range noise, dropouts, intensity variation).
- Increase dataset size and validate on real LiDAR packets.
- If performance becomes a bottleneck, add BVH or GPU acceleration to the
	raycaster, or integrate a higher-fidelity simulator.
