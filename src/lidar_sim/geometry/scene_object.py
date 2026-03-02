from lidar_sim.core.types import ObjectType
from lidar_sim.core.hit import Hit
from lidar_sim.core.ray import Ray
import numpy as np

class SceneObject:
    def __init__(self, object_id: int, object_type: ObjectType):
        self.object_id = object_id
        self.object_type = object_type

    def intersect(self, ray: Ray) -> Hit:
        raise NotImplementedError

    def intersect_batch(self, origins: np.ndarray, directions: np.ndarray):
        """
        origins:    (N, 3) array of ray origins
        directions: (N, 3) array of ray directions
        returns:    (N,) array of distances, inf if no hit
                    (N, 3) array of hit positions
                    (N, 3) array of hit normals
                    (N,) bool array of hits
        """
        raise NotImplementedError
