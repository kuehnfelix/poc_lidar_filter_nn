from __future__ import annotations

import numpy as np


class Ray:
    def __init__(self, origin, direction):

        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)

        if self.origin.shape != (3,):
            raise ValueError("origin must be a 3-element vector")

        if self.direction.shape != (3,):
            raise ValueError("direction must be a 3-element vector")

        norm = np.linalg.norm(self.direction)
        if norm == 0.0:
            raise ValueError("ray direction must be non-zero")
        self.direction /= norm

    def point_at(self, t: float) -> np.ndarray:
        """
        Return the point along the ray at parameter ``t``.
        """
        return self.origin + t * self.direction

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin!r}, direction={self.direction!r})"
