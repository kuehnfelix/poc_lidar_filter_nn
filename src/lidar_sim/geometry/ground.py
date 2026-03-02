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
        hits, positions, normals, mask = self.intersect_batch(
            ray.origin[np.newaxis], ray.direction[np.newaxis]
        )
        if not mask[0]:
            return Hit(False, 0)
        return Hit(True, hits[0], positions[0], normals[0], self.object_id, self.object_type)

    def intersect_batch(self, origins: np.ndarray, directions: np.ndarray):
        EPS = 1e-3
        N = len(origins)
        distances = np.full(N, np.inf)
        positions = np.zeros((N, 3))
        normals = np.zeros((N, 3))

        dz = directions[:, 2]
        valid = np.abs(dz) >= EPS

        t = np.where(valid, (self.z - origins[:, 2]) / np.where(valid, dz, 1.0), -1.0)
        valid &= t >= EPS

        positions[valid] = origins[valid] + t[valid, np.newaxis] * directions[valid]
        distances[valid] = t[valid]
        normals[valid] = [0.0, 0.0, 1.0]

        return distances, positions, normals, valid
