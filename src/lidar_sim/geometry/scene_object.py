from lidar_sim.core.types import ObjectType
from lidar_sim.core.hit import Hit
from lidar_sim.core.ray import Ray

class SceneObject:
    def __init__(self, object_id: int, object_type: ObjectType):
        self.object_id = object_id
        self.object_type = object_type

    def intersect(self, ray: Ray) -> Hit:
        raise NotImplementedError
