import numpy as np
from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit


class Scene:
    def __init__(self):
        self.objects: list[SceneObject] = []
        self.next_object_id = 1
        self.vehicle_pose = None

    def add_object(self, obj: SceneObject):
        obj.object_id = self.next_object_id
        self.next_object_id += 1
        self.objects.append(obj)

    def intersect_many_arrays(self, origins: np.ndarray, directions: np.ndarray) -> list[Hit]:
        """Core batched intersect. Takes raw (N, 3) arrays, returns N Hits."""
        N = len(origins)
        best_distances = np.full(N, np.inf)
        best_positions = np.zeros((N, 3))
        best_normals   = np.zeros((N, 3))
        best_obj_id    = np.full(N, -1, dtype=int)
        best_obj_type  = np.full(N, -1, dtype=int)

        for obj in self.objects:
            distances, positions, normals, mask = obj.intersect_batch(origins, directions)
            closer = mask & (distances < best_distances)
            best_distances[closer] = distances[closer]
            best_positions[closer] = positions[closer]
            best_normals[closer]   = normals[closer]
            best_obj_id[closer]    = obj.object_id
            best_obj_type[closer]  = int(obj.object_type)

        results = []
        for i in range(N):
            if best_obj_id[i] < 0:
                results.append(Hit(False))
            else:
                results.append(Hit(
                    True,
                    best_distances[i],
                    best_positions[i],
                    best_normals[i],
                    best_obj_id[i],
                    best_obj_type[i],
                ))
        return results

    def intersect_many(self, rays: list[Ray]) -> list[Hit]:
        origins    = np.stack([r.origin    for r in rays])
        directions = np.stack([r.direction for r in rays])
        return self.intersect_many_arrays(origins, directions)

    def intersect(self, ray: Ray) -> Hit:
        return self.intersect_many_arrays(
            ray.origin[np.newaxis], ray.direction[np.newaxis]
        )[0]