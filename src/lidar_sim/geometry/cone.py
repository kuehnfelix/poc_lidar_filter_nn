import numpy as np

from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.core.types import ObjectType

class ConeObject(SceneObject):
    def __init__(self, object_id, position, height, radius_base):
        super().__init__(object_id, ObjectType.CONE)
        self.position = position
        self.height = height
        self.radius = radius_base

    def intersect(self, ray: Ray) -> Hit:
        EPS = 1e-3
        ro = ray.origin - self.position
        rd = ray.direction

        k = self.radius / self.height
        k2 = k * k

        # Flip cone: use (h - z)
        ro_z = self.height - ro[2]
        rd_z = -rd[2]

        a = rd[0]**2 + rd[1]**2 - k2 * rd_z**2
        b = 2 * (ro[0]*rd[0] + ro[1]*rd[1] - k2 * ro_z*rd_z)
        c = ro[0]**2 + ro[1]**2 - k2 * ro_z**2

        disc = b*b - 4*a*c
        if disc < 0:
            return Hit(False)

        sqrt_disc = np.sqrt(disc)
        t0 = (-b - sqrt_disc) / (2*a)
        t1 = (-b + sqrt_disc) / (2*a)

        for t in sorted((t0, t1)):
            if t <= EPS:
                continue

            z = ro[2] + t * rd[2]

            # Now valid region is 0 <= z <= height
            if 0.0 <= z <= self.height:
                pos = ray.origin + t * ray.direction

                # Correct normal for flipped cone
                normal = np.array([
                    pos[0] - self.position[0],
                    pos[1] - self.position[1],
                    k2 * (self.height - (pos[2] - self.position[2]))
                ])

                normal /= np.linalg.norm(normal)

                return Hit(True, t, pos, normal,
                        self.object_id, self.object_type)

        return Hit(False)
