"""
Scan pattern that uses real recorded az/el coordinates from .npz frames,
ignoring the real distances (those will be replaced by ray casting).

Invalid points (radius=0 or NaN) are replaced by the nearest valid az/el
in the same packet so every point index always has a usable direction.

Usage in DatasetGenerator:
    from real_recorded_scan_pattern import RealRecordedScanPattern
    scan_pattern = RealRecordedScanPattern("dataset/record2")
    lidar_model  = LiDARModel(scan_pattern=scan_pattern)
"""

import glob
import math
import os
import numpy as np
from typing import Iterator, Tuple
from lidar_sim.lidar.scan_pattern import ScanPattern


class RealRecordedScanPattern(ScanPattern):
    """
    Loads all recorded .npz frames and uses their az/el as scan directions.
    Invalid points are filled with nearest-neighbour interpolation.
    Each call to __iter__ picks a random frame and yields its az/el values
    block by block (matching the 25-block × 5-channel LiDARModel structure).
    """

    def __init__(self, dataset_dir: str):
        paths = sorted(glob.glob(
            os.path.join(dataset_dir, "**/*.npz"), recursive=True
        ))
        if not paths:
            raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

        self.frames = []   # list of (600, 125, 2) — az, el in radians

        n_invalid_total = 0
        n_points_total  = 0

        for path in paths:
            f    = np.load(path)
            data = f["data"]                      # (600, 125, 3)
            az   = data[:, :, 0]                  # (600, 125)
            el   = data[:, :, 1]                  # (600, 125)
            dist = data[:, :, 2]                  # (600, 125)

            n_invalid_total += ((dist == 0) | np.isnan(dist)).sum()
            n_points_total  += dist.size

            self.frames.append(np.stack([az, el], axis=-1))  # (600, 125, 2)

        print(f"RealRecordedScanPattern: loaded {len(self.frames)} frames "
              f"from {dataset_dir}")
        print(f"  Invalid points found: "
              f"{n_invalid_total:,} / {n_points_total:,} "
              f"({n_invalid_total/n_points_total:.1%})")

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        """
        Pick a random recorded frame and yield (az_rad, el_rad) for each
        of the 15,000 blocks (600 packets × 25 blocks).

        LiDARModel calls next() once per block and pairs the result with
        its 5 beam channel offsets to produce 5 rays.

        Since the real LiDAR embeds all 5 beams in each 125-point packet
        (indices 0-4 = block 0, 5-9 = block 1, ...), we yield the mean
        az/el of each block's 5 points as the block centre.
        """
        frame = self.frames[np.random.randint(len(self.frames))]
        # frame: (600, 125, 2)

        output = []

        for pkt_idx in range(600):
            for block_idx in range(25):
                output = []
                for ch_idx in range(5):
                    point = frame[pkt_idx, block_idx + ch_idx*25]
                    output.append(point)

                yield output
