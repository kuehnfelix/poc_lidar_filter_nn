"""Playground for testing LiDAR model and visualization."""
import numpy as np
import matplotlib.pyplot as plt

from lidar_sim.lidar.lidar_model import LiDARModel
from lidar_sim.lidar.elliptic_scan_pattern import EllipticScanPattern
from lidar_sim.lidar.zig_zag_scan_pattern import ZigZagScanPattern
from lidar_sim.scene.scene_generator import SceneGenerator
from lidar_sim.scene.track import Track
from lidar_sim.utils.visualization import LidarVisualizer, visualize_hits



if __name__ == "__main__":
    
    model = LiDARModel(scan_pattern=ZigZagScanPattern())
    sceneGenerator = SceneGenerator(None, None)
    scene = sceneGenerator.generate_scene()
    
    x, y, car_angle = sceneGenerator.track.pose_at_arc_length(0)

    z = 1
    
    lidar_pose = sceneGenerator.sample_lidar_pose(progress_along_track=0.0)
    
    
    all_hits = []
    
    while True:
        try:
            hits = model.measure_single(scene, lidar_pose)
        except StopIteration:
            print("Scan pattern exhausted.")
            break
        
        all_hits.extend(hits)
        
    
    # Option 1: Quick visualization with convenience function
    # Uncomment to visualize:
    # visualize_hits(all_hits, scene_objects=scene.objects)
    
    # Option 2: Interactive visualizer with updates
    # Uncomment to use:
    viz = LidarVisualizer(show_hits=True, show_scene=True, point_size=8.0)
    viz.set_hits(all_hits)
    # viz.set_scene(scene.objects)
    viz.show()
    
