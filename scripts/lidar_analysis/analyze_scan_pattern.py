"""
Analyzes recorded .npz frames to check if the scan pattern
(azimuth/elevation per point index) is consistent across frames.

Usage:
    python analyze_scan_pattern.py --dataset dataset/track_00
    python analyze_scan_pattern.py --dataset dataset/track_00 --plot
"""

import argparse
import glob
import numpy as np
import os


def load_frames(dataset_dir: str):
    """Load all frames from a directory, return (N_frames, 600, 125, 3) array."""
    paths = sorted(glob.glob(os.path.join(dataset_dir, "**/*.npz"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

    frames = []
    for path in paths:
        f = np.load(path)
        frames.append(f["data"])  # (600, 125, 3)

    print(f"Loaded {len(frames)} frames from {dataset_dir}")
    return np.stack(frames), paths  # (N, 600, 125, 3)


def analyze(dataset_dir: str, plot: bool = False):
    frames, paths = load_frames(dataset_dir)
    # frames: (N_frames, 600, 125, 3)
    # axis 3: [azimuth_rad, elevation_rad, distance]

    N_frames, N_packets, N_points, _ = frames.shape

    az  = frames[:, :, :, 0]  # (N, 600, 125)
    el  = frames[:, :, :, 1]  # (N, 600, 125)

    # Use first frame as reference scan pattern
    ref_az = az[0]   # (600, 125)
    ref_el = el[0]   # (600, 125)

    print(f"\nReference frame: {paths[0]}")
    print(f"Frames: {N_frames}  |  Packets/frame: {N_packets}  |  Points/packet: {N_points}")

    # ── Per-frame deviation ────────────────────────────────────────────────────
    az_diff = az - ref_az[np.newaxis]   # (N, 600, 125)
    el_diff = el - ref_el[np.newaxis]   # (N, 600, 125)

    # only compare non-zero points (zero = missed return, not a real measurement)
    valid = frames[:, :, :, 2] > 0     # (N, 600, 125) — distance > 0

    print("\n── Per-frame stats (vs reference frame) ──────────────────────────────")
    print(f"{'Frame':<8} {'Az mean':>10} {'Az std':>10} {'Az max':>10} "
          f"{'El mean':>10} {'El std':>10} {'El max':>10} {'Valid pts':>10}")

    frame_az_means = []
    frame_el_means = []

    for i in range(N_frames):
        v = valid[i]
        if v.sum() == 0:
            continue
        ad = np.abs(az_diff[i][v])
        ed = np.abs(el_diff[i][v])
        frame_az_means.append(ad.mean())
        frame_el_means.append(ed.mean())
        print(f"{i:<8} "
              f"{np.degrees(ad.mean()):>9.4f}° "
              f"{np.degrees(ad.std()):>9.4f}° "
              f"{np.degrees(ad.max()):>9.4f}° "
              f"{np.degrees(ed.mean()):>9.4f}° "
              f"{np.degrees(ed.std()):>9.4f}° "
              f"{np.degrees(ed.max()):>9.4f}° "
              f"{v.sum():>10,}")

    # ── Per-point-index deviation across all frames ────────────────────────────
    # For each of the 125 point positions in each packet, how much does az/el vary?
    print("\n── Per-point-index deviation (across all frames, all packets) ─────────")

    # flatten frames and packets: (N*600, 125, 3)
    flat = frames.reshape(-1, N_points, 3)
    flat_valid = flat[:, :, 2] > 0       # (N*600, 125)

    az_flat = flat[:, :, 0]              # (N*600, 125)
    el_flat = flat[:, :, 1]              # (N*600, 125)

    # per point index: std across all packets that had a valid return
    az_std_per_point = np.zeros(N_points)
    el_std_per_point = np.zeros(N_points)
    az_range_per_point = np.zeros(N_points)
    el_range_per_point = np.zeros(N_points)

    for pt in range(N_points):
        v = flat_valid[:, pt]
        if v.sum() < 2:
            continue
        az_std_per_point[pt]   = np.degrees(az_flat[v, pt].std())
        el_std_per_point[pt]   = np.degrees(el_flat[v, pt].std())
        az_range_per_point[pt] = np.degrees(az_flat[v, pt].ptp())
        el_range_per_point[pt] = np.degrees(el_flat[v, pt].ptp())

    print(f"Azimuth  std  — mean over points: {az_std_per_point.mean():.4f}°  "
          f"max: {az_std_per_point.max():.4f}°  "
          f"worst point idx: {az_std_per_point.argmax()}")
    print(f"Azimuth  range— mean over points: {az_range_per_point.mean():.4f}°  "
          f"max: {az_range_per_point.max():.4f}°")
    print(f"Elevation std  — mean over points: {el_std_per_point.mean():.4f}°  "
          f"max: {el_std_per_point.max():.4f}°  "
          f"worst point idx: {el_std_per_point.argmax()}")
    print(f"Elevation range— mean over points: {el_range_per_point.mean():.4f}°  "
          f"max: {el_range_per_point.max():.4f}°")

    # ── Verdict ────────────────────────────────────────────────────────────────
    print("\n── Verdict ────────────────────────────────────────────────────────────")
    az_threshold = 0.1   # degrees
    el_threshold = 0.1

    az_ok = az_std_per_point.mean() < az_threshold
    el_ok = el_std_per_point.mean() < el_threshold

    if az_ok and el_ok:
        print("✓ Scan pattern is consistent across frames — safe to use as fixed pattern.")
    else:
        if not az_ok:
            print(f"✗ Azimuth varies significantly (mean std={az_std_per_point.mean():.4f}° "
                  f"> threshold {az_threshold}°)")
        if not el_ok:
            print(f"✗ Elevation varies significantly (mean std={el_std_per_point.mean():.4f}° "
                  f"> threshold {el_threshold}°)")
        print("  The scan pattern is NOT fixed — point index alone does not encode a stable az/el.")
        print("  Your model should rely on the az/el feature values, not their position in the packet.")

    # ── Optional plot ──────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.suptitle("Scan Pattern Consistency Analysis")

            axes[0, 0].plot(np.degrees(frame_az_means))
            axes[0, 0].set_title("Mean |az deviation| per frame (deg)")
            axes[0, 0].set_xlabel("Frame")
            axes[0, 0].set_ylabel("Degrees")

            axes[0, 1].plot(np.degrees(frame_el_means))
            axes[0, 1].set_title("Mean |el deviation| per frame (deg)")
            axes[0, 1].set_xlabel("Frame")
            axes[0, 1].set_ylabel("Degrees")

            axes[1, 0].plot(az_std_per_point)
            axes[1, 0].set_title("Azimuth std per point index (deg)")
            axes[1, 0].set_xlabel("Point index (0-124)")
            axes[1, 0].set_ylabel("Std (deg)")

            axes[1, 1].plot(el_std_per_point)
            axes[1, 1].set_title("Elevation std per point index (deg)")
            axes[1, 1].set_xlabel("Point index (0-124)")
            axes[1, 1].set_ylabel("Std (deg)")

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/record2",
                        help="Directory containing .npz frame files")
    parser.add_argument("--plot", action="store_true",
                        help="Show matplotlib plots of deviation")
    args = parser.parse_args()

    analyze(args.dataset, plot=args.plot)
