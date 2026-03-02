import numpy as np
from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.core.types import ObjectType

class ConeObject(SceneObject):
    def __init__(self, object_id, position, height, radius_base):
        super().__init__(object_id, ObjectType.CONE)
        self.position = np.array(position, dtype=np.float64)
        self.height = float(height)
        self.radius = float(radius_base)

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

        k = self.radius / self.height
        k2 = k * k

        ro = origins - self.position          # (N, 3)
        rd = directions                        # (N, 3)

        ro_z = self.height - ro[:, 2]         # (N,)
        rd_z = -rd[:, 2]                       # (N,)

        a = rd[:, 0]**2 + rd[:, 1]**2 - k2 * rd_z**2
        b = 2.0 * (ro[:, 0]*rd[:, 0] + ro[:, 1]*rd[:, 1] - k2 * ro_z*rd_z)
        c = ro[:, 0]**2 + ro[:, 1]**2 - k2 * ro_z**2

        disc = b*b - 4.0*a*c
        valid_disc = disc >= 0.0

        sqrt_disc = np.where(valid_disc, np.sqrt(np.maximum(disc, 0.0)), 0.0)
        a_safe = np.where(np.abs(a) > 1e-12, a, 1.0)

        t0 = (-b - sqrt_disc) / (2.0 * a_safe)
        t1 = (-b + sqrt_disc) / (2.0 * a_safe)
        t_near = np.minimum(t0, t1)
        t_far  = np.maximum(t0, t1)

        # try near root first, fall back to far
        for t_cand in (t_near, t_far):
            remaining = valid_disc & ~mask & (t_cand > EPS)
            if not np.any(remaining):
                continue
            z = ro[:, 2] + t_cand * rd[:, 2]
            in_cone = remaining & (z >= 0.0) & (z <= self.height)
            if np.any(in_cone):
                t = t_cand[in_cone]
                pos = origins[in_cone] + t[:, np.newaxis] * directions[in_cone]
                local = pos - self.position
                nx = local[:, 0]
                ny = local[:, 1]
                nz = k2 * (self.height - local[:, 2])
                norm = np.sqrt(nx**2 + ny**2 + nz**2)
                norm = np.where(norm > 0, norm, 1.0)
                normals[in_cone] = np.stack([nx/norm, ny/norm, nz/norm], axis=1)
                positions[in_cone] = pos
                distances[in_cone] = t
                mask[in_cone] = True

        return distances, positions, normals, mask
