import numpy as np
from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.core.types import ObjectType

class CylinderObject(SceneObject):
    def __init__(self, object_id, base_center, radius, height):
        super().__init__(object_id, ObjectType.CYLINDER)
        self.base = np.array(base_center, dtype=np.float64)
        self.radius = float(radius)
        self.height = float(height)

    def intersect(self, ray: Ray) -> Hit:
        distances, positions, normals, mask = self.intersect_batch(
            ray.origin[np.newaxis], ray.direction[np.newaxis]
        )
        if not mask[0]:
            return Hit(False)
        return Hit(True, distances[0], positions[0], normals[0], self.object_id, self.object_type)

    def intersect_batch(self, origins: np.ndarray, directions: np.ndarray):
        EPS = 1e-6
        N = len(origins)
        distances = np.full(N, np.inf)
        positions = np.zeros((N, 3))
        normals = np.zeros((N, 3))
        mask = np.zeros(N, dtype=bool)

        ro = origins - self.base   # (N, 3)
        rd = directions            # (N, 3)

        # --- side ---
        a = rd[:, 0]**2 + rd[:, 1]**2
        valid_a = a >= EPS
        b = 2.0 * (ro[:, 0]*rd[:, 0] + ro[:, 1]*rd[:, 1])
        c = ro[:, 0]**2 + ro[:, 1]**2 - self.radius**2
        disc = b**2 - 4.0*a*np.where(valid_a, c, 0.0)
        valid_disc = valid_a & (disc >= 0.0)

        sqrt_disc = np.where(valid_disc, np.sqrt(np.maximum(disc, 0.0)), 0.0)
        a_safe = np.where(valid_a, a, 1.0)
        t0 = (-b - sqrt_disc) / (2.0 * a_safe)
        t1 = (-b + sqrt_disc) / (2.0 * a_safe)

        for t_cand in (np.minimum(t0, t1), np.maximum(t0, t1)):
            remaining = valid_disc & ~mask & (t_cand > EPS)
            z = ro[:, 2] + t_cand * rd[:, 2]
            hit = remaining & (z >= 0.0) & (z <= self.height)
            if np.any(hit):
                t = t_cand[hit]
                pos = origins[hit] + t[:, np.newaxis] * rd[hit]
                n = pos - self.base
                n[:, 2] = 0.0
                n /= np.linalg.norm(n, axis=1, keepdims=True)
                distances[hit] = t
                positions[hit] = pos
                normals[hit] = n
                mask[hit] = True

        # --- bottom cap (z=0 in local space) ---
        valid_rd = np.abs(rd[:, 2]) > EPS
        t_bot = np.where(valid_rd, -ro[:, 2] / np.where(valid_rd, rd[:, 2], 1.0), -1.0)
        hit_bot = valid_rd & (t_bot > EPS) & (t_bot < distances)
        pos_bot = origins[hit_bot] + t_bot[hit_bot, np.newaxis] * rd[hit_bot]
        in_cap = ((pos_bot[:, 0]-self.base[0])**2 + (pos_bot[:, 1]-self.base[1])**2) <= self.radius**2
        idx = np.where(hit_bot)[0][in_cap]
        distances[idx] = t_bot[idx]
        positions[idx] = pos_bot[in_cap]
        normals[idx] = [0.0, 0.0, -1.0]
        mask[idx] = True

        # --- top cap ---
        t_top = np.where(valid_rd, (self.height - ro[:, 2]) / np.where(valid_rd, rd[:, 2], 1.0), -1.0)
        hit_top = valid_rd & (t_top > EPS) & (t_top < distances)
        pos_top = origins[hit_top] + t_top[hit_top, np.newaxis] * rd[hit_top]
        in_cap = ((pos_top[:, 0]-self.base[0])**2 + (pos_top[:, 1]-self.base[1])**2) <= self.radius**2
        idx = np.where(hit_top)[0][in_cap]
        distances[idx] = t_top[idx]
        positions[idx] = pos_top[in_cap]
        normals[idx] = [0.0, 0.0, 1.0]
        mask[idx] = True

        return distances, positions, normals, mask
