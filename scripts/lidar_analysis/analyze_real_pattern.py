"""
Analyzes the real LiDAR scan pattern by looking at az/el values
across packets and checking for periodicity/repetition.

Usage:
    python analyze_real_pattern.py --dataset dataset/record2
    python analyze_real_pattern.py --dataset dataset/record2 --plot
"""

import argparse
import glob
import os
import numpy as np


def load_all_packets(dataset_dir: str):
    """Load all packets as flat sequence, preserving order. Returns (N, 125, 3)."""
    paths = sorted(glob.glob(os.path.join(dataset_dir, "**/*.npz"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

    all_packets = []
    for path in paths:
        f = np.load(path)
        all_packets.append(f["data"])  # (600, 125, 3)

    data = np.concatenate(all_packets, axis=0)  # (N_total, 125, 3)
    print(f"Loaded {len(data):,} packets from {len(paths)} frames in {dataset_dir}")
    return data


def analyze(dataset_dir: str, plot: bool = False):
    packets = load_all_packets(dataset_dir)  # (N, 125, 3)
    N = len(packets)

    # Use first point of each packet as the "phase" indicator
    # since it's the leading edge of the scan window
    az_first = np.degrees(packets[:, 0, 0])   # (N,) azimuth of first point
    el_first = np.degrees(packets[:, 0, 1])   # (N,) elevation of first point

    # Also look at mean az/el per packet to characterise the window
    valid = packets[:, :, 2] > 0              # (N, 125)
    az_mean = np.array([
        np.degrees(packets[i, valid[i], 0]).mean() if valid[i].any() else np.nan
        for i in range(N)
    ])
    el_mean = np.array([
        np.degrees(packets[i, valid[i], 1]).mean() if valid[i].any() else np.nan
        for i in range(N)
    ])

    # ── Az/El range of the sensor ─────────────────────────────────────────────
    all_az = np.degrees(packets[:, :, 0][valid])
    all_el = np.degrees(packets[:, :, 1][valid])

    print(f"\n── Sensor az/el range ────────────────────────────────────────────────")
    print(f"Azimuth:   min={all_az.min():.2f}°  max={all_az.max():.2f}°  "
          f"range={all_az.ptp():.2f}°")
    print(f"Elevation: min={all_el.min():.2f}°  max={all_el.max():.2f}°  "
          f"range={all_el.ptp():.2f}°")

    # ── Find scan period: how many packets until az repeats? ──────────────────
    print(f"\n── Scan period detection ─────────────────────────────────────────────")

    # Look for the period by finding where az_first returns to ~same value
    # Use autocorrelation of az_mean to find the dominant period
    az_centered = az_mean - np.nanmean(az_mean)
    az_centered = np.nan_to_num(az_centered)

    # autocorrelation
    max_lag = min(N // 2, 2000)
    autocorr = np.correlate(az_centered[:max_lag*2], az_centered[:max_lag*2], mode='full')
    autocorr = autocorr[len(autocorr)//2:]   # keep positive lags
    autocorr /= autocorr[0]                  # normalise

    # find peaks in autocorrelation (excluding lag=0)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(autocorr[1:max_lag], height=0.3, distance=10)
    peaks += 1  # offset for lag=0 exclusion

    if len(peaks) > 0:
        period = peaks[0]
        print(f"Dominant scan period: {period} packets")
        print(f"At 6000 Hz that's {period/6000*1000:.2f}ms per scan cycle")
        print(f"Peaks at lags: {peaks[:10].tolist()}")
    else:
        period = None
        print("No clear periodic pattern found in autocorrelation")

    # ── Check if az/el within a period are consistent ─────────────────────────
    if period is not None:
        print(f"\n── Consistency within period ({period} packets) ──────────────────────")

        # Align packets to period and check repeatability
        n_complete = (N // period) * period
        shaped = az_mean[:n_complete].reshape(-1, period)  # (n_cycles, period)

        cycle_std  = shaped.std(axis=0)    # (period,) — variation at each phase
        cycle_mean = shaped.mean(axis=0)   # (period,) — mean pattern

        print(f"Complete cycles found: {n_complete // period}")
        print(f"Az variation at same phase — mean std: {cycle_std.mean():.4f}°  "
              f"max std: {cycle_std.max():.4f}°  "
              f"at packet offset: {cycle_std.argmax()}")

        # same for elevation
        el_shaped = el_mean[:n_complete].reshape(-1, period)
        el_cycle_std = el_shaped.std(axis=0)
        print(f"El variation at same phase — mean std: {el_cycle_std.mean():.4f}°  "
              f"max std: {el_cycle_std.max():.4f}°")

        if cycle_std.mean() < 0.5:
            print(f"\n✓ Scan pattern repeats consistently every {period} packets")
            print(f"  You can model it as a fixed {period}-packet cycle")
        else:
            print(f"\n~ Pattern repeats but with some jitter ({cycle_std.mean():.2f}° std)")

    # ── Show the scan pattern for one cycle ───────────────────────────────────
    if period is not None:
        print(f"\n── Pattern for first cycle (packet 0 to {period-1}) ─────────────────")
        print(f"{'Pkt':>5}  {'Az mean':>10}  {'El mean':>10}  {'Az first':>10}  {'El first':>10}")
        for i in range(min(period, 20)):
            print(f"{i:>5}  {az_mean[i]:>9.3f}°  {el_mean[i]:>9.3f}°  "
                  f"{az_first[i]:>9.3f}°  {el_first[i]:>9.3f}°")
        if period > 20:
            print(f"  ... ({period - 20} more packets)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 2, figsize=(16, 12))
            fig.suptitle("Real LiDAR Scan Pattern Analysis")

            # az/el of first point over time
            axes[0, 0].plot(az_first[:600], lw=0.8)
            axes[0, 0].set_title("Azimuth of first point per packet (first 600)")
            axes[0, 0].set_xlabel("Packet index")
            axes[0, 0].set_ylabel("Degrees")

            axes[0, 1].plot(el_first[:600], lw=0.8, color='orange')
            axes[0, 1].set_title("Elevation of first point per packet (first 600)")
            axes[0, 1].set_xlabel("Packet index")
            axes[0, 1].set_ylabel("Degrees")

            # mean az/el per packet
            axes[1, 0].plot(az_mean[:600], lw=0.8)
            axes[1, 0].set_title("Mean azimuth per packet (first 600)")
            axes[1, 0].set_xlabel("Packet index")
            axes[1, 0].set_ylabel("Degrees")

            axes[1, 1].plot(el_mean[:600], lw=0.8, color='orange')
            axes[1, 1].set_title("Mean elevation per packet (first 600)")
            axes[1, 1].set_xlabel("Packet index")
            axes[1, 1].set_ylabel("Degrees")

            # autocorrelation
            axes[2, 0].plot(autocorr[:max_lag], lw=0.8)
            if len(peaks) > 0:
                axes[2, 0].axvline(period, color='red', linestyle='--',
                                   label=f'period={period}')
                axes[2, 0].legend()
            axes[2, 0].set_title("Autocorrelation of mean azimuth")
            axes[2, 0].set_xlabel("Lag (packets)")
            axes[2, 0].set_ylabel("Correlation")

            # az vs el scatter (scan pattern shape)
            sample = min(N * 125, 50000)
            az_flat = all_az[::max(1, len(all_az)//sample)]
            el_flat = all_el[::max(1, len(all_el)//sample)]
            axes[2, 1].scatter(az_flat, el_flat, s=0.5, alpha=0.3)
            axes[2, 1].set_title("Az vs El scatter (scan pattern shape)")
            axes[2, 1].set_xlabel("Azimuth (deg)")
            axes[2, 1].set_ylabel("Elevation (deg)")

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib/scipy not installed — skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/record2")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    analyze(args.dataset, plot=args.plot)
