import numpy as np

from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.scene.scene import Scene
from lidar_sim.lidar.scan_pattern import ScanPattern

class LiDARModel:
    ch1_angle = np.deg2rad(-48)
    ch2_angle = np.deg2rad(-24)
    ch3_angle = np.deg2rad(0.0)
    ch4_angle = np.deg2rad(24)
    ch5_angle = np.deg2rad(48)
    channels = [ch1_angle, ch2_angle, ch3_angle, ch4_angle, ch5_angle]

    def __init__(self, scan_pattern: ScanPattern):
        self.scan_pattern = iter(scan_pattern)
    
    def measure_single(self, scene: Scene, vehicle_pose) -> list[Hit]:
        """Generate rays for a single scan and return the resulting hits."""
        rays = self.generate_rays(vehicle_pose)
        hits = [scene.intersect(ray) for ray in rays]
        return hits

    def generate_rays(self, vehicle_pose) -> list[Ray]:
        """Return a list of rays expressed in world coordinates.

        Parameters
        ----------
        vehicle_pose : array-like, shape (4,4)
            Homogeneous transform mapping points from the LiDAR sensor frame to
            the world frame.

        Returns
        -------
        list[Ray]
            Ray instances originating at the vehicle/lidar position and
            pointing outward along the specified pattern in world coordinates.
        """

        # convert pose to numpy array for ease of use
        pose = np.asarray(vehicle_pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("vehicle_pose must be a 4x4 homogeneous matrix")

        origin = pose[:3, 3]
        rotation = pose[:3, :3]

        rays: list[Ray] = []
        
        azimuth, elevation = next(self.scan_pattern)
        
        for channel in self.channels:
            az = azimuth + channel
            el = elevation
        
            x = np.cos(el) * np.cos(az)
            y = np.cos(el) * np.sin(az)
            z = np.sin(el)
            local_dir = np.array([x, y, z], dtype=float)
            
            world_dir = rotation @ local_dir
            rays.append(Ray(origin, az, el, world_dir))

        return rays
