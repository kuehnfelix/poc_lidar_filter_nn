from lidar_sim.scene.scene import Scene
from lidar_sim.core.types import ObjectType
import numpy as np

class LiDARSimulator:
    def __init__(self, lidar_model):
        self.lidar = lidar_model

    def scan(self, scene: Scene):
        features = []
        labels = []

        rays = self.lidar.generate_rays(scene.vehicle_pose)
        for ray in rays:
            hit = scene.intersect(ray)
            features.append(self._make_features(ray, hit))
            labels.append(hit.object_type == ObjectType.CONE)
        return np.array(features), np.array(labels)
