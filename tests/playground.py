"""Playground for testing LiDAR model and visualization."""
import numpy as np

from lidar_sim.lidar.lidar_model import LiDARModel
from lidar_sim.lidar.elliptic_scan_pattern import EllipticScanPattern
from lidar_sim.scene.scene_generator import SceneGenerator
from lidar_sim.utils.visualization import LidarVisualizer, visualize_hits



if __name__ == "__main__":
    model = LiDARModel(scan_pattern=EllipticScanPattern())
    sceneGenerator = SceneGenerator(None, None)
    scene = sceneGenerator.generate_scene()
    
    
    lidar_pose = np.eye(4)
    lidar_pose[2, 3] = 1.2
    # Rotate 10 degrees down around y-axis
    angle = np.radians(10)
    lidar_pose[0, 0] = np.cos(angle)
    lidar_pose[0, 2] = np.sin(angle)
    lidar_pose[2, 0] = -np.sin(angle)
    lidar_pose[2, 2] = np.cos(angle)
    
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
    viz = LidarVisualizer(show_hits=True, show_scene=True, point_size=4.0)
    viz.set_hits(all_hits)
    # viz.set_scene(scene.objects)
    viz.show()
    
