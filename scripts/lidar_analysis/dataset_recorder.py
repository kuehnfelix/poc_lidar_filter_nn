"""
Receives MSOP packets over UDP, parses them, and saves them in the
dataset format expected by LiDARPacketDataset.

Each MSOP packet = 25 blocks x 5 channels = 125 points = 1 dataset packet.
600 consecutive packets (by sequence number) are grouped into one frame.

Dropped or out-of-order packets are detected via the 2-byte sequence number
at bytes 4-6 of the header. If a packet is dropped, the frame is discarded
and recording starts fresh from the next packet.

Output layout:
    dataset/
        track_00/
            frame_000.npz   # data: (600, 125, 3), labels: (600, 125) all zeros
            frame_001.npz

Usage:
    python record_dataset.py --output dataset/track_00 --frames 100
"""

import socket
import struct
import math
import numpy as np
import argparse
import os
import time

from collections import deque

MSOP_PORT = 6699
MSOP_PACKET_SIZE = 1210
HEADER_SIZE = 32
DATA_BLOCK_COUNT = 25
DATA_BLOCK_SIZE = 47
CHANNEL_COUNT = 5
PACKETS_PER_FRAME = 600
SEQ_MAX = 65536   # 2-byte sequence number wraps at 65536


def parse_seq(data: bytes) -> int:
    """Extract 2-byte sequence number from bytes 4-6 of the header."""
    return struct.unpack(">H", data[4:6])[0]


def seq_next(seq: int) -> int:
    """Next expected sequence number, wrapping at 65536."""
    return (seq + 1) % SEQ_MAX


def parse_channel(channel_bytes):
    radius_raw = struct.unpack(">H", channel_bytes[0:2])[0]
    elev_raw = struct.unpack(">H", channel_bytes[2:4])[0]
    azim_raw = struct.unpack(">H", channel_bytes[4:6])[0]
    radius_m = radius_raw * 0.005
    elevation_deg = (elev_raw - 32768) * 0.01
    azimuth_deg = (azim_raw - 32768) * 0.01
    return radius_m, elevation_deg, azimuth_deg


def parse_msop_packet(data: bytes):
    """
    Returns (seq, points) where points is (125, 3) float32 [az_rad, el_rad, dist].
    Returns (seq, None) if packet is malformed.
    """
    if len(data) != MSOP_PACKET_SIZE:
        return None, None

    seq = parse_seq(data)
    points = np.zeros((DATA_BLOCK_COUNT * CHANNEL_COUNT, 3), dtype=np.float32)
    offset = HEADER_SIZE
    data = data[offset:]
    idx = 0

    for ch_idx in range(CHANNEL_COUNT):
        ch_offset = 2 + ch_idx * 9
        for pkt_idx in range(DATA_BLOCK_COUNT):
            packet_offset = pkt_idx * DATA_BLOCK_SIZE
            base_idx = packet_offset + ch_offset
            
            radius_raw = struct.unpack(">H", data[base_idx:base_idx+2])[0]
            elev_raw = struct.unpack(">H", data[base_idx+2:base_idx+4])[0]
            azim_raw = struct.unpack(">H", data[base_idx+4:base_idx+6])[0]
            
            radius_m = radius_raw * 0.005
            elevation_deg = (elev_raw - 32768) * 0.01
            azimuth_deg = (azim_raw - 32768) * 0.01
            
            points[ch_idx * DATA_BLOCK_COUNT + pkt_idx] = (np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg), radius_m)            

    return seq, points


def record(output_dir: str, num_frames: int):
    os.makedirs(output_dir, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", MSOP_PORT))
    print(f"Listening on UDP port {MSOP_PORT}...")
    print(f"Recording {num_frames} frames x {PACKETS_PER_FRAME} packets → {output_dir}")

    frame_idx = 0
    packet_queue = deque()  # queue of (seq, points)
    seq_set = set()         # quick lookup for duplicates
    expected_seq = None
    dropped_total = 0
    start_time = time.time()

    while frame_idx < num_frames:
        data, _ = sock.recvfrom(MSOP_PACKET_SIZE)
        seq, points = parse_msop_packet(data)
        if points is None:
            continue

        # initialize expected sequence
        if expected_seq is None:
            expected_seq = seq

        # skip duplicates
        if seq in seq_set:
            continue

        # insert into queue in order
        packet_queue.append((seq, points))
        seq_set.add(seq)

        # sort queue if needed (rare in practice)
        packet_queue = deque(sorted(packet_queue, key=lambda x: x[0]))

        # remove old packets in case queue grows too large
        while len(packet_queue) > PACKETS_PER_FRAME * 2:
            old_seq, _ = packet_queue.popleft()
            seq_set.remove(old_seq)
            dropped_total += 1

        # check if we can build a frame
        if len(packet_queue) >= PACKETS_PER_FRAME:
            # take first PACKETS_PER_FRAME packets
            frame_packets = [p[1] for p in list(packet_queue)[:PACKETS_PER_FRAME]]
            data_arr = np.stack(frame_packets)
            labels_arr = np.zeros((PACKETS_PER_FRAME, 125), dtype=np.uint8)

            path = os.path.join(output_dir, f"frame_{frame_idx:03d}.npz")
            np.savez(path, data=data_arr, labels=labels_arr)

            elapsed = time.time() - start_time
            print(f"Saved frame {frame_idx+1}/{num_frames} → {path} ({elapsed:.1f}s elapsed)")

            # remove used packets from queue and set
            for i in range(PACKETS_PER_FRAME):
                old_seq, _ = packet_queue.popleft()
                seq_set.remove(old_seq)

            frame_idx += 1
            # update expected_seq to next packet after last used
            if packet_queue:
                expected_seq = (packet_queue[0][0] - PACKETS_PER_FRAME) % SEQ_MAX
            else:
                expected_seq = None

    sock.close()
    print(f"\nDone. {num_frames} frames saved to {output_dir}")
    if dropped_total:
        print(f"Total dropped packets: {dropped_total}")
    else:
        print("No dropped packets detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/record2")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    record(args.output, args.frames)