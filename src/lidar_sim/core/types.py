from enum import IntEnum

class ObjectType(IntEnum):
    GROUND = 0
    CONE = 1
    CYLINDER = 2
    BOX = 3
    FENCE = 4
    VEHICLE = 5
    UNKNOWN = 255
