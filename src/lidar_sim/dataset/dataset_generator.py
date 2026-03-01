import os

import numpy as np

from lidar_sim.core.hit import Hit
from lidar_sim.core.ray import Ray
from lidar_sim.core.types import ObjectType
from lidar_sim.scene.scene_generator import SceneGenerator
from lidar_sim.lidar.lidar_model import LiDARModel
from lidar_sim.lidar.zig_zag_scan_pattern import ZigZagScanPattern
from lidar_sim.scene.track import Track

class DatasetGenerator:
    def __init__(self, 
                 scene_generator = SceneGenerator(), 
                 lidar_model = LiDARModel(scan_pattern=ZigZagScanPattern()),
                 dataset_path = "dataset"):
        self.scene_generator = scene_generator
        self.scene = scene_generator.generate_scene()
        self.lidar_model = lidar_model
        self.dataset_path = dataset_path
        
    def generate_dataset(self, num_tracks=2, frames_per_track=100):
        for track_idx in range(num_tracks):
            print(f"Generating track {track_idx+1}/{num_tracks}...")
            self.new_scene()
            track = self.scene_generator.track
            for frame_idx in range(frames_per_track):
                progress = frame_idx / frames_per_track * track.total_spline_length()
                print(f"  Generating frame {frame_idx+1}/{frames_per_track} (progress: {progress:.2f}m)...")
                data_packets, label_packets = self.generate_one_frame(progress)
                self.save_frame(track_idx, frame_idx, data_packets, label_packets)

    def save_frame(self, track_idx, frame_idx, data_packets, label_packets):
        track_dir = f"{self.dataset_path}/track_{track_idx:02d}"
        os.makedirs(track_dir, exist_ok=True)
        frame_path = f"{track_dir}/frame_{frame_idx:03d}.npz"
        np.savez(frame_path, data=data_packets, labels=label_packets)

    def new_scene(self):
        self.scene = self.scene_generator.generate_scene()

    def generate_one_frame(self, progress_along_track, scan_pattern=ZigZagScanPattern()):
        self.lidar_model.scan_pattern = iter(scan_pattern)
        lateral_error = np.random.normal(0, 0.5) 
        angular_error = np.random.normal(0, np.deg2rad(20))
        height_error = np.random.normal(0, 0.2)
        height = 1.0 + height_error
        lidar_angle_error = np.random.normal(0, np.deg2rad(3))
        lidar_angle = np.radians(10) + lidar_angle_error
        
        lidar_pose = self.scene_generator.sample_lidar_pose(progress_along_track, lateral_error, angular_error, height, lidar_angle)
        
        data_packets = []
        label_packets = []
        
        for _ in range(600):
            packet_data, packet_labels = self.generate_one_packet(progress_along_track, lidar_pose)
            data_packets.append(packet_data)
            label_packets.append(packet_labels)
            
        return data_packets, label_packets
        
        
    def generate_one_packet(self, progress_along_track, lidar_pose):
        
        packet_data = np.zeros((125, 3), dtype=np.float32)
        packet_labels = np.zeros((125), dtype=np.uint8)
        for i in range(25):
            hits, rays = self.lidar_model.measure_single(self.scene, lidar_pose)            
            for k, (hit, ray) in enumerate(zip(hits, rays)):
                if hit.hit:
                    index = k * 25 + i
                    packet_data[index, :] = (ray.azimuth, ray.elevation, hit.distance)
                    if hit.object_type == ObjectType.CONE:
                        packet_labels[index] = 1        
        
        return packet_data, packet_labels
    
    
if __name__ == "__main__":
    generator = DatasetGenerator()
    generator.generate_dataset(num_tracks=2, frames_per_track=100)