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
        ... # TODO
