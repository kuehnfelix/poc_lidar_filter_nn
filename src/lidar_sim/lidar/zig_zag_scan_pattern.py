"""Scan pattern implementations for LIDAR simulation."""
import math
from typing import Iterator, Tuple
from lidar_sim.lidar.scan_pattern import ScanPattern


class ZigZagScanPattern(ScanPattern):
    """Scan pattern following an elliptic curve."""

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        for x in range(15000):
            azimuth = 25/math.pi * math.acos(math.cos(math.pi * x * math.pi / 120)) - 12.5
            elevation = 12.5 - (25 * x / 15000)

            yield (
                math.radians(azimuth),
                math.radians(elevation),
            )