import os
import time
import numpy as np
from lidar_sim.core.hit import Hit
from lidar_sim.core.ray import Ray
from lidar_sim.core.types import ObjectType
from lidar_sim.scene.scene_generator import SceneGenerator
from lidar_sim.lidar.lidar_model import LiDARModel
from lidar_sim.lidar.real_recorded_scan_pattern import RealRecordedScanPattern
from lidar_sim.scene.track import Track


class DatasetGenerator:
    N_STEPS   = 15000   # scan steps per frame
    N_CH      = 5       # channels per step
    N_PACKETS = 600     # packets per frame
    N_SCANS   = 25      # scan steps per packet
    # sanity: N_PACKETS * N_SCANS == N_STEPS  (600 * 25 = 15000)

    def __init__(self,
                 scene_generator=SceneGenerator(),
                 lidar_model=LiDARModel(scan_pattern=RealRecordedScanPattern("recorded_lidar_packets")),
                 dataset_path="dataset"):
        self.scene_generator = scene_generator
        self.scene           = scene_generator.generate_scene()
        self.lidar_model     = lidar_model
        self.dataset_path    = dataset_path

    def generate_dataset(self, num_tracks=2, frames_per_track=100):
        start_time = time.time()
        for track_idx in range(7, num_tracks):
            print(f"Generating track {track_idx+1}/{num_tracks}...")
            self.new_scene()
            track = self.scene_generator.track
            for frame_idx in range(frames_per_track):
                progress = frame_idx / frames_per_track * track.total_spline_length()
                print(f"  Frame {frame_idx+1}/{frames_per_track} "
                      f"(progress: {progress:.2f}m, "
                      f"elapsed: {time.time()-start_time:.2f}s)...")
                data_packets, label_packets = self.generate_one_frame(progress)
                self.save_frame(track_idx, frame_idx, data_packets, label_packets)

    def save_frame(self, track_idx, frame_idx, data_packets, label_packets):
        track_dir = f"{self.dataset_path}/track_{track_idx:02d}"
        os.makedirs(track_dir, exist_ok=True)
        np.savez(f"{track_dir}/frame_{frame_idx:03d}.npz",
                 data=data_packets, labels=label_packets)

    def new_scene(self):
        self.scene = self.scene_generator.generate_scene()

    def generate_one_frame(self, progress_along_track):
        lateral_error  = np.random.normal(0, 0.3)
        angular_error  = np.random.normal(0, np.deg2rad(10))
        height_error   = np.random.normal(0, 0.1)
        height         = 1.0 + height_error
        lidar_angle    = np.radians(10) + np.random.normal(0, np.deg2rad(2))

        lidar_pose = self.scene_generator.sample_lidar_pose(
            progress_along_track, lateral_error, angular_error, height, lidar_angle
        )

        # fire all 75000 rays at once
        hits_flat, rays_flat = self.lidar_model.measure_frame(self.scene, lidar_pose)

        data_packets  = np.zeros((self.N_PACKETS, 125, 3),  dtype=np.float32)
        label_packets = np.zeros((self.N_PACKETS, 125),     dtype=np.uint8)

        for packet_idx in range(self.N_PACKETS):
            for scan_idx in range(self.N_SCANS):
                flat_base = (packet_idx * self.N_SCANS + scan_idx) * self.N_CH
                for ch_idx in range(self.N_CH):
                    flat_i  = flat_base + ch_idx
                    store_i = ch_idx * self.N_SCANS + scan_idx  # matches original indexing

                    hit = hits_flat[flat_i]
                    ray = rays_flat[flat_i]

                    distance = 0
                    if hit.hit:
                        distance = hit.distance * np.random.normal(1, 1.0005) # add some noise to the distance measurement
                        label_packets[packet_idx, store_i] = 1 if hit.object_type == ObjectType.CONE else 0
                        
                    data_packets[packet_idx, store_i]  = (ray.azimuth, ray.elevation, distance)

        return data_packets, label_packets


if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate_dataset(num_tracks=30, frames_per_track=200)