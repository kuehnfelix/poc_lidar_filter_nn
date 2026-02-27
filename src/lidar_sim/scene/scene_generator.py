
import numpy as np
from lidar_sim.scene.scene import Scene
from lidar_sim.geometry.ground import GroundPlane
from lidar_sim.geometry.box import BoxObject

from lidar_sim.scene.track import Track, TrackGenerationError, Color
from lidar_sim.geometry.cone import ConeObject
from lidar_sim.geometry.cylinder import CylinderObject


class SceneGenerator:
    def __init__(self, track_generator, obstacle_config):
        self.track_generator = track_generator
        self.obstacle_config = obstacle_config

    def generate_scene(self) -> Scene:
        scene = Scene()
        self._add_ground(scene)
        self._add_track_cones(scene)
        #self._add_obstacles(scene)
        self._sample_vehicle_pose(scene)
        return scene

# TODO: Implement the following methods
    def _add_ground(self, scene: Scene):
        scene.add_object(GroundPlane())
        
    def _add_track_cones(self, scene: Scene):
        track = Track()
        track.generate_random_track()
        
        for c in track.cones:
            
            height, radius = (.505, 0.1425) if c.color == Color.ORANGE else (.325, 0.114)
            
            distance = np.linalg.norm([c.x, c.y])
            if distance > 10:
                continue
            cone = ConeObject(-1, np.array([c.x, c.y, 0]), height, radius)
            scene.add_object(cone)

    
    def _add_obstacles(self, scene: Scene):
        for i in range(10):
            x = np.random.uniform(0, 20)
            y = np.random.uniform(-10, 10)
            z = 0.5
            
            center=np.array([x, y, z])
            size = np.ones(3)
            orientation = np.eye(3)
            
            box = BoxObject(i, center, size, orientation)
            scene.add_object(box)
            
    def _sample_vehicle_pose(self, scene: Scene):
        pass  # --- IGNORE ---
