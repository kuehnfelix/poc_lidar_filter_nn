import numpy as np

from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.core.types import ObjectType

class CylinderObject(SceneObject):
    def __init__(self, object_id, base_center, radius, height):
        super().__init__(object_id, ObjectType.CYLINDER)
        self.base = base_center
        self.radius = radius
        self.height = height

    def intersect(self, ray: Ray) -> Hit:
        EPS = 1e-6

        ro = ray.origin - self.base
        rd = ray.direction

        a = rd[0]**2 + rd[1]**2

        if abs(a) < EPS:
            return Hit(False)

        b = 2 * (ro[0]*rd[0] + ro[1]*rd[1])
        c = ro[0]**2 + ro[1]**2 - self.radius**2

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
            if 0.0 <= z <= self.height:
                pos = ray.origin + t * ray.direction
                normal = np.array([pos[0]-self.base[0],
                                pos[1]-self.base[1],
                                0.0])
                normal /= np.linalg.norm(normal)
                return Hit(True, t, pos, normal, self.object_id, self.object_type)

        return Hit(False)
  
