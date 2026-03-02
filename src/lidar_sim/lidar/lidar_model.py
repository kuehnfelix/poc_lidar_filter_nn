import numpy as np

from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.scene.scene import Scene
from lidar_sim.lidar.scan_pattern import ScanPattern
from lidar_sim.lidar.zig_zag_scan_pattern import ZigZagScanPattern


class LiDARModel:
    ch1_angle = np.deg2rad(-48)
    ch2_angle = np.deg2rad(-24)
    ch3_angle = np.deg2rad(0.0)
    ch4_angle = np.deg2rad(24)
    ch5_angle = np.deg2rad(48)
    channels = np.array([ch1_angle, ch2_angle, ch3_angle, ch4_angle, ch5_angle])  # (5,)

    def __init__(self, scan_pattern: ScanPattern):
        self.scan_pattern = iter(scan_pattern)

    def measure_single(self, scene: Scene, vehicle_pose) -> tuple:
        """5 rays for a single azimuth/elevation step."""
        rays = self.generate_rays(vehicle_pose)
        origins    = np.stack([r.origin    for r in rays])
        directions = np.stack([r.direction for r in rays])
        hits = scene.intersect_many_arrays(origins, directions)
        return hits, rays

    def measure_frame(self, scene: Scene, vehicle_pose) -> tuple:
        """All 15000 scan steps x 5 channels = 75000 rays, fired at once."""
        pose = np.asarray(vehicle_pose, dtype=float)
        origin   = pose[:3, 3]
        rotation = pose[:3, :3]

        pattern    = list(ZigZagScanPattern())
        azimuths   = np.array([p[0] for p in pattern])   # (15000,)
        elevations = np.array([p[1] for p in pattern])   # (15000,)

        # (15000, 5) — broadcast channels across all scan steps
        az = azimuths[:, np.newaxis] + self.channels[np.newaxis, :]
        el = elevations[:, np.newaxis] * np.ones((1, 5))

        # local directions (15000, 5, 3)
        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        local_dirs = np.stack([x, y, z], axis=-1)

        # rotate to world (15000, 5, 3)
        world_dirs = local_dirs @ rotation.T

        # flatten to (75000, 3)
        N_total = 15000 * 5
        origins_flat    = np.broadcast_to(origin, (N_total, 3)).copy()
        directions_flat = world_dirs.reshape(-1, 3)

        # single intersect call for all 75000 rays
        hits_flat = scene.intersect_many_arrays(origins_flat, directions_flat)

        az_flat = az.reshape(-1)
        el_flat = el.reshape(-1)
        rays = [
            Ray(origin, az_flat[i], el_flat[i], directions_flat[i])
            for i in range(N_total)
        ]

        return hits_flat, rays

    def generate_rays(self, vehicle_pose) -> list[Ray]:
        pose = np.asarray(vehicle_pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("vehicle_pose must be a 4x4 homogeneous matrix")

        origin   = pose[:3, 3]
        rotation = pose[:3, :3]

        azimuth, elevation = next(self.scan_pattern)
        az = azimuth + self.channels   # (5,)
        el = np.full(5, elevation)

        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        local_dirs = np.stack([x, y, z], axis=1)  # (5, 3)
        world_dirs = local_dirs @ rotation.T       # (5, 3)

        return [Ray(origin, az[i], el[i], world_dirs[i]) for i in range(5)]