import numpy as np
from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.core.types import ObjectType

class BoxObject(SceneObject):
    def __init__(self, object_id, center, size, orientation):
        super().__init__(object_id, ObjectType.BOX)
        self.center = np.array(center, dtype=np.float64)
        self.size = np.array(size, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)

    def intersect(self, ray: Ray) -> Hit:
        distances, positions, normals, mask = self.intersect_batch(
            ray.origin[np.newaxis], ray.direction[np.newaxis]
        )
        if not mask[0]:
            return Hit(False)
        return Hit(True, distances[0], positions[0], normals[0], self.object_id, self.object_type)

    def intersect_batch(self, origins: np.ndarray, directions: np.ndarray):
        EPS = 1e-3
        N = len(origins)
        distances = np.full(N, np.inf)
        positions = np.zeros((N, 3))
        normals = np.zeros((N, 3))
        mask = np.zeros(N, dtype=bool)

        invR = self.orientation.T                               # (3, 3)
        ro = (origins - self.center) @ invR.T                  # (N, 3)
        rd = directions @ invR.T                               # (N, 3)

        half = self.size * 0.5                                  # (3,)

        tmin = np.full(N, -np.inf)
        tmax = np.full(N,  np.inf)
        valid = np.ones(N, dtype=bool)
        hit_axis = np.zeros(N, dtype=int)
        hit_sign = np.zeros(N)

        for i in range(3):
            rd_i = rd[:, i]
            ro_i = ro[:, i]
            parallel = np.abs(rd_i) < EPS
            outside = parallel & ((ro_i < -half[i]) | (ro_i > half[i]))
            valid &= ~outside

            safe_rd = np.where(parallel, 1.0, rd_i)
            t1 = (-half[i] - ro_i) / safe_rd
            t2 = ( half[i] - ro_i) / safe_rd
            t_near = np.minimum(t1, t2)
            t_far  = np.maximum(t1, t2)

            # track which axis set tmin (for normal)
            update = ~parallel & (t_near > tmin)
            hit_axis = np.where(update, i, hit_axis)
            hit_sign = np.where(update, np.sign(rd_i), hit_sign)  # ray enters from which side

            tmin = np.maximum(tmin, np.where(parallel, tmin, t_near))
            tmax = np.minimum(tmax, np.where(parallel, tmax, t_far))
            valid &= tmin <= tmax

        hit = valid & (tmin > EPS)
        if np.any(hit):
            t = tmin[hit]
            pos = origins[hit] + t[:, np.newaxis] * directions[hit]
            local_hit = ro[hit] + t[:, np.newaxis] * rd[hit]

            # normal: largest component of local_hit / half
            abs_local = np.abs(local_hit) / half
            axis = np.argmax(abs_local, axis=1)
            n_local = np.zeros_like(local_hit)
            n_local[np.arange(len(axis)), axis] = np.sign(local_hit[np.arange(len(axis)), axis])
            n_world = n_local @ self.orientation.T
            n_world /= np.linalg.norm(n_world, axis=1, keepdims=True)

            distances[hit] = t
            positions[hit] = pos
            normals[hit] = n_world
            mask[hit] = True

        return distances, positions, normals, mask
