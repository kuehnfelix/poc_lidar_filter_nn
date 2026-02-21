import numpy as np

class DatasetGenerator:
    def __init__(self, scene_generator, lidar_simulator):
        self.scene_generator = scene_generator
        self.lidar_sim = lidar_simulator

    def generate(self, num_samples):
        X, y = [], []
        for _ in range(num_samples):
            scene = self.scene_generator.generate_scene()
            features, labels = self.lidar_sim.scan(scene)
            X.append(features)
            y.append(labels)
        return np.stack(X), np.stack(y)
