"""Ray tracing primitives for LiDAR simulation."""

from __future__ import annotations

import numpy as np


class Ray:
    """Represents a ray with origin and direction in 3D space."""
    
    def __init__(self, origin, azimuth, elevation, direction):
        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.azimuth = azimuth
        self.elevation = elevation

        if self.origin.shape != (3,):
            raise ValueError("origin must be a 3-element vector")

        if self.direction.shape != (3,):
            raise ValueError("direction must be a 3-element vector")

        norm = np.linalg.norm(self.direction)
        if norm == 0.0:
            raise ValueError("ray direction must be non-zero")
        self.direction /= norm

    def point_at(self, t: float) -> np.ndarray:
        """Return the point along the ray at parameter ``t``."""
        return self.origin + t * self.direction
