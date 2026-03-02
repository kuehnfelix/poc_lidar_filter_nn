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


def _fix_invalid(az: np.ndarray, el: np.ndarray, dist: np.ndarray):
    """
    Replace invalid points (dist==0 or NaN in az/el) with linear interpolation
    along the point axis within each packet.

    az, el, dist: (N_packets, 125)
    returns: az, el with invalids filled
    """
    invalid = (dist == 0) | np.isnan(az) | np.isnan(el) | \
              np.isnan(dist) | (dist < 0)

    az = az.copy()
    el = el.copy()
    indices = np.arange(125)

    for pkt_idx in range(az.shape[0]):
        inv = invalid[pkt_idx]
        if not inv.any():
            continue
        if inv.all():
            continue  # entire packet invalid — leave as-is

        valid_idx = np.where(~inv)[0]

        # np.interp does linear interp and clamps at edges (nearest extrapolation)
        az[pkt_idx] = np.interp(indices, valid_idx, az[pkt_idx, valid_idx])
        el[pkt_idx] = np.interp(indices, valid_idx, el[pkt_idx, valid_idx])

    return az, el


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

            az_fixed, el_fixed = _fix_invalid(az, el, dist)
            self.frames.append(np.stack([az_fixed, el_fixed], axis=-1))  # (600, 125, 2)

        print(f"RealRecordedScanPattern: loaded {len(self.frames)} frames "
              f"from {dataset_dir}")
        print(f"  Invalid points fixed: "
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

        for pkt_idx in range(600):
            for block_idx in range(25):
                pts = frame[pkt_idx, block_idx*5 : block_idx*5+5]  # (5, 2)
                az_centre = float(pts[:, 0].mean())
                el_centre = float(pts[:, 1].mean())
                yield (az_centre, el_centre)
