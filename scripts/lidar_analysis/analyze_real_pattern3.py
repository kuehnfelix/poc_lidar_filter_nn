"""
Scan pattern analysis for MEMS LiDAR with spiral/lissajous pattern.
Finds the full scan period and visualizes the 2D pattern.

Usage:
    python analyze_real_pattern3.py --dataset dataset/record2 --plot
"""

import argparse
import glob
import os
import numpy as np


def load_all_packets(dataset_dir: str):
    paths = sorted(glob.glob(os.path.join(dataset_dir, "**/*.npz"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")
    frames = [np.load(p)["data"] for p in paths]
    data = np.concatenate(frames, axis=0)  # (N, 125, 3)
    print(f"Loaded {len(data):,} packets from {len(paths)} frames")
    return data


def analyze(dataset_dir: str, plot: bool = False):
    packets = load_all_packets(dataset_dir)  # (N, 125, 3)
    N = len(packets)

    az   = np.degrees(packets[:, :, 0])  # (N, 125)
    el   = np.degrees(packets[:, :, 1])  # (N, 125)
    dist = packets[:, :, 2]
    valid = dist > 0

    all_az = az[valid]
    all_el = el[valid]

    print(f"\n── Sensor FOV ────────────────────────────────────────────────────────")
    print(f"Azimuth:   {all_az.min():.2f}° to {all_az.max():.2f}°  "
          f"(range {all_az.ptp():.2f}°)")
    print(f"Elevation: {all_el.min():.2f}° to {all_el.max():.2f}°  "
          f"(range {all_el.ptp():.2f}°)")

    # ── Mean az and el per packet ─────────────────────────────────────────────
    pkt_az = np.array([az[i][valid[i]].mean() if valid[i].any() else np.nan for i in range(N)])
    pkt_el = np.array([el[i][valid[i]].mean() if valid[i].any() else np.nan for i in range(N)])

    # ── Find period using elevation ───────────────────────────────────────────
    # Elevation goes top-to-bottom in ~one frame, so it's more monotonic
    # and easier to find the reset point than azimuth
    el_clean = np.nan_to_num(pkt_el, nan=np.nanmean(pkt_el))

    # find where elevation resets (jumps back to top)
    el_diff = np.diff(el_clean)
    el_diff_std = el_diff.std()

    # a reset is a large positive jump (bottom back to top)
    # threshold: 3 std above mean
    reset_threshold = el_diff.mean() + 3 * el_diff_std
    resets = np.where(el_diff > reset_threshold)[0] + 1

    print(f"\n── Frame period detection via elevation resets ───────────────────────")
    if len(resets) > 1:
        periods = np.diff(resets)
        period  = int(np.median(periods))
        print(f"Found {len(resets)} elevation resets")
        print(f"Packets per frame: min={periods.min()}  max={periods.max()}  "
              f"median={period}  std={periods.std():.1f}")
        print(f"At 6000 Hz: {period/6000*1000:.1f}ms per frame  "
              f"= {6000/period:.2f} Hz frame rate")
        print(f"Reset positions (first 10): {resets[:10].tolist()}")
    else:
        period = 600  # fallback
        print(f"Could not detect resets automatically — using default {period}")

    # ── Analyze one clean frame ───────────────────────────────────────────────
    # find a clean frame (no resets inside it)
    print(f"\n── Single frame scan pattern ({period} packets × 125 points) ─────────")

    # pick frame starting at first reset
    if len(resets) > 0:
        frame_start = resets[0]
    else:
        frame_start = 0

    frame_end = frame_start + period
    if frame_end > N:
        print("Not enough data for a full frame after first reset")
        frame_end = N

    frame_az = az[frame_start:frame_end][valid[frame_start:frame_end]]
    frame_el = el[frame_start:frame_end][valid[frame_start:frame_end]]

    print(f"Points in one frame: {len(frame_az):,}")
    print(f"Az: {frame_az.min():.2f}° to {frame_az.max():.2f}°")
    print(f"El: {frame_el.min():.2f}° to {frame_el.max():.2f}°")

    # ── Az and El oscillation frequencies ────────────────────────────────────
    # In a Lissajous pattern az and el oscillate at different frequencies
    # find the ratio by counting zero crossings
    az_centered = pkt_az - np.nanmean(pkt_az)
    el_centered = pkt_el - np.nanmean(pkt_el)
    az_centered = np.nan_to_num(az_centered)
    el_centered = np.nan_to_num(el_centered)

    az_crossings = np.where(np.diff(np.sign(az_centered)))[0]
    el_crossings = np.where(np.diff(np.sign(el_centered)))[0]

    if len(az_crossings) > 1 and len(el_crossings) > 1:
        # average half-period in packets
        az_half_period = np.median(np.diff(az_crossings))
        el_half_period = np.median(np.diff(el_crossings))
        print(f"\n── Oscillation analysis ──────────────────────────────────────────────")
        print(f"Az zero-crossings: {len(az_crossings)}  "
              f"half-period: {az_half_period:.1f} packets  "
              f"= {az_half_period/6000*1000:.2f}ms  "
              f"= {6000/(az_half_period*2):.1f} Hz")
        print(f"El zero-crossings: {len(el_crossings)}  "
              f"half-period: {el_half_period:.1f} packets  "
              f"= {el_half_period/6000*1000:.2f}ms  "
              f"= {6000/(el_half_period*2):.1f} Hz")
        if el_half_period > 0:
            ratio = az_half_period / el_half_period
            print(f"Az/El frequency ratio: {ratio:.3f}  "
                  f"(Lissajous ratio ≈ {round(ratio*2)/2:.1f})")

    # ── Check if 5 beams have consistent offsets ──────────────────────────────
    print(f"\n── Beam spacing (point index % 5) ───────────────────────────────────")
    # within a packet, the 5 points of each block are the 5 beams
    # check their relative az/el offsets
    for beam in range(5):
        beam_idx = np.arange(beam, 125, 5)  # indices 0,5,10... or 1,6,11... etc.
        b_az = az[:, beam_idx][valid[:, beam_idx]]
        b_el = el[:, beam_idx][valid[:, beam_idx]]
        if len(b_az):
            print(f"  Beam {beam}: az={b_az.mean():+.3f}°±{b_az.std():.3f}°  "
                  f"el={b_el.mean():+.3f}°±{b_el.std():.3f}°")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(18, 12))
            fig.suptitle("MEMS LiDAR Scan Pattern Analysis")

            # full scatter coloured by time
            ax1 = fig.add_subplot(2, 3, 1)
            n_sample = min(len(all_az), 80000)
            idx = np.linspace(0, len(all_az)-1, n_sample, dtype=int)
            sc = ax1.scatter(all_az[idx], all_el[idx], c=idx, s=0.5,
                             cmap='rainbow', alpha=0.4)
            plt.colorbar(sc, ax=ax1, label='time')
            ax1.set_title("Az vs El (coloured by time)")
            ax1.set_xlabel("Azimuth (deg)")
            ax1.set_ylabel("Elevation (deg)")

            # one clean frame only
            ax2 = fig.add_subplot(2, 3, 2)
            frame_data = packets[frame_start:frame_end]
            frame_valid = frame_data[:, :, 2] > 0
            faz = np.degrees(frame_data[:, :, 0][frame_valid])
            fel = np.degrees(frame_data[:, :, 1][frame_valid])
            # colour by packet index within frame to show sweep direction
            pkt_color = np.repeat(np.arange(period),
                                  frame_valid.sum(axis=1))[:len(faz)]
            ax2.scatter(faz, fel, c=pkt_color[:len(faz)], s=0.8,
                        cmap='viridis', alpha=0.5)
            ax2.set_title(f"One frame ({period} pkts, coloured by pkt order)")
            ax2.set_xlabel("Azimuth (deg)")
            ax2.set_ylabel("Elevation (deg)")

            # mean az over time
            ax3 = fig.add_subplot(2, 3, 3)
            ax3.plot(pkt_az[:period*3], lw=0.7, label='az')
            ax3.plot(pkt_el[:period*3], lw=0.7, label='el', alpha=0.7)
            for r in resets[resets < period*3]:
                ax3.axvline(r, color='red', alpha=0.4, lw=0.5)
            ax3.set_title("Mean az/el per packet (3 frames)")
            ax3.set_xlabel("Packet index")
            ax3.set_ylabel("Degrees")
            ax3.legend()

            # az over time (zoom one frame)
            ax4 = fig.add_subplot(2, 3, 4)
            ax4.plot(pkt_az[frame_start:frame_end], lw=0.8)
            ax4.set_title("Az per packet — one frame")
            ax4.set_xlabel("Packet within frame")
            ax4.set_ylabel("Mean azimuth (deg)")

            # el over time (zoom one frame)
            ax5 = fig.add_subplot(2, 3, 5)
            ax5.plot(pkt_el[frame_start:frame_end], lw=0.8, color='orange')
            ax5.set_title("El per packet — one frame")
            ax5.set_xlabel("Packet within frame")
            ax5.set_ylabel("Mean elevation (deg)")

            # beam offsets
            ax6 = fig.add_subplot(2, 3, 6)
            for beam in range(5):
                beam_idx = np.arange(beam, 125, 5)
                b_az = az[frame_start:frame_end, beam_idx]
                b_el = el[frame_start:frame_end, beam_idx]
                bv   = (frame_data[:, beam_idx, 2] > 0)
                ax6.scatter(b_az[bv], b_el[bv], s=0.5, alpha=0.4,
                            label=f'beam {beam}')
            ax6.set_title("Beams separated (one frame)")
            ax6.set_xlabel("Azimuth (deg)")
            ax6.set_ylabel("Elevation (deg)")
            ax6.legend(markerscale=5)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset/record2")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    analyze(args.dataset, plot=args.plot)
