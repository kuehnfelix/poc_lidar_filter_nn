import numpy as np

from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.core.types import ObjectType

class GroundPlane(SceneObject):
    

    def __init__(self, z=0.0):
        super().__init__(object_id=0, object_type=ObjectType.GROUND)
        self.z = z

    def intersect(self, ray: Ray) -> Hit:
        EPS = 1e-3
        
        dz = ray.direction[2]
        if abs(dz) < EPS:
            return Hit(False)

        distance = (self.z - ray.origin[2]) / dz # distance along ray to intersection

        if distance < EPS: # Filter out hits behind the ray origin or too close
            return Hit(False)

        pos = ray.origin + distance * ray.direction
        normal = np.array([0.0, 0.0, 1.0])

        return Hit(True, distance, pos, normal, self.object_id, self.object_type)


