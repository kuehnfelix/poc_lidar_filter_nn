import numpy as np

from lidar_sim.geometry.scene_object import SceneObject
from lidar_sim.core.ray import Ray
from lidar_sim.core.hit import Hit
from lidar_sim.core.types import ObjectType

class BoxObject(SceneObject):
    def __init__(self, object_id, center, size, orientation):
        super().__init__(object_id, ObjectType.BOX)
        self.center = center
        self.size = size            # (x,y,z)
        self.orientation = orientation  # rotation matrix

    def intersect(self, ray: Ray) -> Hit:
        EPS = 1e-3

        invR = self.orientation.T
        ro = invR @ (ray.origin - self.center)
        rd = invR @ ray.direction

        half = self.size * 0.5
        tmin, tmax = -np.inf, np.inf

        for i in range(3):
            if abs(rd[i]) < EPS:
                if ro[i] < -half[i] or ro[i] > half[i]:
                    return Hit(False)
            else:
                t1 = (-half[i] - ro[i]) / rd[i]
                t2 = ( half[i] - ro[i]) / rd[i]
                t_near, t_far = min(t1, t2), max(t1, t2)
                tmin = max(tmin, t_near)
                tmax = min(tmax, t_far)
                if tmin > tmax:
                    return Hit(False)

        if tmin <= EPS:
            return Hit(False)

        pos = ray.origin + tmin * ray.direction
        local_hit = ro + tmin * rd

        normal_local = np.zeros(3)
        idx = np.argmax(np.abs(local_hit / half))
        normal_local[idx] = np.sign(local_hit[idx])

        normal = self.orientation @ normal_local
        normal /= np.linalg.norm(normal)

        return Hit(True, tmin, pos, normal, self.object_id, self.object_type)
