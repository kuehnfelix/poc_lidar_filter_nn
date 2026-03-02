LiDAR Cone Point Classification — Proof of Concept
=================================================

End-to-end pipeline: simulate → generate data → train → infer on real LiDAR.


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


Components
----------

- [Simulation](docs/simulation.md)
- [Dataset Generation](docs/dataset_generation.md)  
- [Neural Network](docs/neural_network.md)
- [Real LiDAR](docs/real_lidar.md)

Project Structure
-----------------
```
│
├── docs/                           # Documentation
├── recorded_lidar_packets/         # .npz LiDAR frames used by RealRecordedScanPattern
├── scripts/
│   └── lidar_analysis/             # Offline analysis scripts for LiDAR data
│       └── pcaps/                  # Raw PCAP captures from the physical LiDAR sensor
├── src/
│   ├── lidar_sim/                  # LiDAR simulation library
│   │   ├── core/                   # Shared primitives: Ray, Hit, types
│   │   ├── dataset/                # Dataset generation (DatasetGenerator)
│   │   ├── geometry/               # Scene objects: cone, box, cylinder, ground plane
│   │   ├── lidar/                  # LiDAR model and scan patterns
│   │   ├── scene/                  # Scene, SceneGenerator, Track
│   │   └── utils/                  # Shared utilities
│   └── neural_net/                 # Neural network for cone classification
│       ├── configs/                # Model/training hyperparameter configs
│       └── models/                 # Model definitions
└── tests/                          # Unit and integration tests
```

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
- Stored in ML-friendly formats (`.npz`).
- Features: shaped `(N, 125, F)` where F are per-point features such as
	distance, azimuth, elevation and derivatives of distance.
- Labels: shaped `(N, 125)`, binary per-point.

Neural network
--------------
- A compact 1-D convolutional network operating on the ordered 125-point
	scanline. Convolutions run along point order and produce per-point outputs.
	This matches the physical scan pattern and keeps latency low for embedded use.


Notes & next steps
------------------
- Add configurable noise models (range noise, dropouts, intensity variation).
- Increase dataset size and validate on real LiDAR packets.
- If performance becomes a bottleneck, add BVH or GPU acceleration to the
	raycaster, or integrate a higher-fidelity simulator.
