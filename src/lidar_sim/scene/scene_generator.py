from lidar_sim.scene.scene import Scene
from lidar_sim.geometry.ground import GroundPlane

class SceneGenerator:
    def __init__(self, track_generator, obstacle_config):
        self.track_generator = track_generator
        self.obstacle_config = obstacle_config

    def generate_scene(self) -> Scene:
        scene = Scene()
        self._add_ground(scene)
        self._add_track_cones(scene)
        self._add_obstacles(scene)
        self._sample_vehicle_pose(scene)
        return scene

# TODO: Implement the following methods
    def _add_ground(self, scene: Scene):
        scene.add_object(GroundPlane())
        
    def _add_track_cones(self, scene: Scene):
        pass  # --- IGNORE ---
    def _add_obstacles(self, scene: Scene):
        pass  # --- IGNORE ---
    def _sample_vehicle_pose(self, scene: Scene):
        pass  # --- IGNORE ---
