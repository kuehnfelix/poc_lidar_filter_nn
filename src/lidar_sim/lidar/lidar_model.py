import numpy as np

from lidar_sim.core.ray import Ray

class LiDARModel:
    azimuth_start_ch1   = -60 * np.pi / 180
    azimuth_end_ch1     = -35 * np.pi / 180
    azimuth_start_ch2   = -36.25 * np.pi / 180
    azimuth_end_ch2     = 11.25 * np.pi / 180
    azimuth_start_ch3   = -12.5 * np.pi / 180
    azimuth_end_ch3     = 12.5 * np.pi / 180
    azimuth_start_ch4   = 11.25 * np.pi / 180
    azimuth_end_ch4     = 36.25 * np.pi / 180
    azimuth_start_ch5   = 35 * np.pi / 180
    azimuth_end_ch5     = 60 * np.pi / 180

    def __init__(self, angles, noise_params):
        self.angles = angles        # list of (az, el)
        self.noise = noise_params

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
        for az, el in self.angles:
            # direction in sensor (vehicle) coordinates
            x = np.cos(el) * np.cos(az)
            y = np.cos(el) * np.sin(az)
            z = np.sin(el)
            local_dir = np.array([x, y, z], dtype=float)
            world_dir = rotation @ local_dir
            rays.append(Ray(origin, world_dir))

        return rays
