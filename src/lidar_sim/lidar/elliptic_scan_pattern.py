"""Scan pattern implementations for LIDAR simulation."""
import math
from typing import Iterator, Tuple
from lidar_sim.lidar.scan_pattern import ScanPattern


class EllipticScanPattern(ScanPattern):
    """Scan pattern following an elliptic curve."""

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        for x in range(15000):
            azimuth = -12.5 * math.cos(math.pi * x / 120)
            elevation = 12.5 - (25 * x / 15000)

            yield (
                math.radians(azimuth),
                math.radians(elevation),
            )