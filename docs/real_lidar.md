# Real LiDAR

## Sensor (measured)

| Property | Value |
|---|---|
| Protocol | MSOP UDP port 6699, 1210 bytes/packet |
| Packet rate | 6000 Hz |
| Frame rate | 10 Hz (600 packets/frame, exact) |
| Beams | 5 at az offsets ±48°, ±24°, 0° |
| Az FOV | ±62° |
| El FOV | ±12° |
| Points/packet | 125 (25 blocks × 5 beams) |
| Scan pattern | Zigzag: az fast ~600 Hz, el slow 10 Hz |
| Invalid sentinel | az = el = -327.68° = -5.7190948 rad |

## Packet Layout
```
[0:4]   header magic
[4:6]   sequence number (uint16 BE, wraps at 65536)
[6:32]  timestamp + metadata
[32:]   25 × 47-byte data blocks:
          [0]    time_offset
          [1]    return_seq  
          [2:47] 5 channels × 9 bytes:
                   radius    uint16 × 0.005 = metres
                   elevation uint16 (val-32768) × 0.01 = degrees
                   azimuth   uint16 (val-32768) × 0.01 = degrees
                   intensity uint8
```

## Analysis Tools
```bash
# scan pattern consistency across frames
python analyze_scan_pattern.py --dataset dataset/record2 --plot

# full pattern analysis: period, beam layout, frequencies
python analyze_real_pattern3.py --dataset dataset/record2 --plot
```

## Key Findings

- Exactly 600 packets/frame, zero jitter
- 5 beams confirmed at ±48°, ±24°, 0° (point_index % 5)
- Hood occlusion: 34 always-invalid positions in packets 596-599
- Gaps in az/el coverage = car body occlusion, not missing data