# Benchmark data

The benchmark consists of **23 real indoor scenes (5,946 frames)** captured
with a mobile multi-sensor platform, spanning bright, dim, and lights-off
illumination. It is derived from our previously published **RadarEyes**
dataset: the capture sessions below are RadarEyes recordings, exported with
the canonical pipeline in this repository (`tools/export_canonical_gt.py`,
deterministic every-8th-frame test split).

**Full benchmark data will be released upon paper acceptance.** Until then
the repository ships (a) the exact 5k RAMI seed clouds used in the paper
(`assets/seeds/<tag>.ply`) and (b) the strict-protocol per-scene results
(`assets/results/per_scene_results.csv`), so all headline numbers can be
inspected without the raw data.

## Scene catalog

LLF = low-light fraction, the mean fraction of GT pixels with luma < 40
(tiers: dark > 35%, dim 15–35%, bright < 15%). Train/test = canonical split.

| Scene | tag | Tier | LLF | Train | Test |
|---|---|---|---|---|---|
| DarkLobby-1 | ndarklob1 | dark | 71% | 314 | 45 |
| DarkCorr | ndarkzou | dark | 56% | 262 | 38 |
| DarkHall | nhall2 | dark | 54% | 157 | 23 |
| DarkLobby-2 | ndarklob2 | dark | 39% | 314 | 45 |
| Corridor-A | n0630c | dim | 34% | 313 | 45 |
| Lounge-A | n2F | dim | 29% | 630 | 90 |
| Dim-C | n0703c | dim | 22% | 123 | 18 |
| Hall-F | nhall | dim | 22% | 158 | 23 |
| Hall-C | n1Fhall | dim | 20% | 158 | 23 |
| Dim-B | n0703b | dim | 20% | 171 | 25 |
| Lounge-B | nyizi | dim | 19% | 158 | 23 |
| Hall-B | A409hall1 | dim | 18% | 262 | 38 |
| Room-A | F2room | dim | 17% | 157 | 23 |
| Dim-A | n0703a | dim | 17% | 83 | 12 |
| Hall-E | nF2hall | dim | 17% | 157 | 23 |
| Dim-D | n0630d | bright | 14% | 313 | 45 |
| Hall-D | nA409h2 | bright | 14% | 260 | 38 |
| Hall-A | F2hall2 | bright | 12% | 157 | 23 |
| Office-C | dark401a | bright | 10% | 157 | 23 |
| Elev-B | nF4dianti | bright | 7% | 158 | 23 |
| Office-B | n0630b | bright | 6% | 314 | 45 |
| Office-A | n0630a | bright | 5% | 314 | 45 |
| Elev-A | nF2dianti | bright | 3% | 105 | 15 |

The tag ↔ capture-directory mapping is `configs/scenes.json`.

## Sensors and per-scene format

Each capture directory contains four modalities used by this code:

```
<capture>/
├── Camera_ZED/            # ZED 2i stereo camera, HD720 (1280x720)
│   ├── frame_*.png        # decoded RGB frames (the benchmark frames)
│   ├── pose.txt           # dense 6-DoF pose stream [x y z qx qy qz qw]
│   └── timestamp.txt      # per-tick timestamps
├── 1843_azi/ADC/          # TI IWR1843 mmWave radar, horizontal mount
│   └── frame_*.bin        # raw complex ADC (samples=128, chirps=128, rx=4, tx=3)
├── 1843_ele/ADC/          # second IWR1843, rotated (vertical) mount
├── Lidar_pcd/             # LiDAR reference (used only by the LiDAR
│   └── frame_*.bin        # control arm and geometry evaluation), float32 (N,4)
└── (each modality has its own timestamp.txt; frames are matched by
    nearest-timestamp lookup at load time)
```

- RGB: locked intrinsics fx = fy = 528.0, cx = 640.0, cy = 360.0,
  right-handed Z-up world frame.
- Radar: single-chip TI IWR1843, ~0.125 m range resolution; this code
  re-processes raw ADC (range FFT + CFAR), no pre-built point clouds are
  used.
- Typical rates: radar 10–20 Hz, RGB (decoded) ~6 Hz, LiDAR ~10 Hz,
  pose ~100 Hz.

## Volume

Measured on the 23 benchmark scenes, used modalities only (RGB frames +
both radar ADC streams + LiDAR + poses): **~100 GB total, ~4.4 GB mean per
scene** (min ~2.0 GB for a 30 s capture, max ~11.5 GB for the 120 s Lounge-A).
The RAMI pipeline itself needs only RGB + poses + the horizontal-radar ADC;
LiDAR is used by the control arm and evaluation only.
