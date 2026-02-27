from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
import numpy as np

class Scene:
    def __init__(self):
        self.objects: list[SceneObject] = []
        self.next_object_id = 1
        self.vehicle_pose = None

    def add_object(self, obj: SceneObject):
        obj.object_id = self.next_object_id
        self.next_object_id += 1
        self.objects.append(obj)

    def intersect(self, ray: Ray) -> Hit:
        closest_hit = Hit(hit=False, distance=np.inf)
        for obj in self.objects:
            hit = obj.intersect(ray)
            if hit.hit and hit.distance < closest_hit.distance:
                closest_hit = hit
        return closest_hit
