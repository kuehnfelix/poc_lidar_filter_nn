"""Playground for testing LiDAR model and visualization."""
import numpy as np
import matplotlib.pyplot as plt

from lidar_sim.lidar.lidar_model import LiDARModel
from lidar_sim.lidar.elliptic_scan_pattern import EllipticScanPattern
from lidar_sim.scene.scene_generator import SceneGenerator



if __name__ == "__main__":
    model = LiDARModel(scan_pattern=EllipticScanPattern())
    sceneGenerator = SceneGenerator(None, None)
    scene = sceneGenerator.generate_scene()
    
    print(scene.objects)


    plt.xlim(-50, 50)
    plt.ylim(0, 100)
    plt.axes().set_aspect('equal', adjustable='box')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("LiDAR Scan Pattern")
    plt.grid()
    
    positions = np.empty((0, 2))
    
    lidar_pose = np.eye(4)
    lidar_pose[2, 3] = 1.0
    # Rotate 10 degrees down around y-axis
    angle = np.radians(10)
    lidar_pose[0, 0] = np.cos(angle)
    lidar_pose[0, 2] = np.sin(angle)
    lidar_pose[2, 0] = -np.sin(angle)
    lidar_pose[2, 2] = np.cos(angle)
    
    while True:
        try:
            hits = model.measure_single(scene, lidar_pose)
        except StopIteration:
            print("Scan pattern exhausted.")
            break
        
        hit_positions = np.array([hit.position for hit in hits if hit.hit])
        
        if hit_positions.size == 0:
            continue
        
        positions = np.vstack((positions, hit_positions[:, :2]))
        
    plt.scatter(positions[:, 1], positions[:, 0], s=1, c='blue', label='Hits')
    plt.legend()
    plt.show()