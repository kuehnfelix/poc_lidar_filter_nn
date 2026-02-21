import numpy as np

class Hit:
    def __init__(self, hit=False, distance=np.inf, position=None, normal=None,
                 object_id=-1, object_type=None):
        self.hit = hit
        self.distance = distance
        self.position = position
        self.normal = normal
        self.object_id = object_id
        self.object_type = object_type
