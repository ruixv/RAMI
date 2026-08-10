"""2D Cell-Averaging CFAR detector and RDAE-cube → detection conversion.

CFAR (Constant False Alarm Rate) thresholds a power map by estimating the
local noise floor from a "training" ring of cells around each cell under
test, with a guard ring between to avoid self-contamination. Detection rule:

    P(CUT) > α · mean(training_cells)
    α = N_train · (PFA^(-1/N_train) - 1)

We use the cell-averaging variant (CA-CFAR) because it is simple, well
characterized, and adequate for a deterministic geometric anchor pipeline.
Sliding-window means are computed via integral image (cumulative sums) so
the cost is O(H · W) regardless of window size.
"""

from __future__ import annotations

import numpy as np

from .rdae import RadarConfig, build_bin_geometry


def ca_cfar_2d(
    power: np.ndarray,
    train_size: tuple[int, int] = (8, 4),
    guard_size: tuple[int, int] = (4, 2),
    pfa: float = 1e-3,
) -> np.ndarray:
    """Run 2D CA-CFAR on a power map.

    Args:
        power: (H, W) non-negative float array (e.g. magnitude squared).
        train_size: half-extent of the training ring per axis. The full ring
            covers `2 * (train + guard) + 1` cells, minus the inner
            `2 * guard + 1` guard region.
        guard_size: half-extent of the guard ring per axis.
        pfa: probability of false alarm. Smaller → fewer false detections,
            also fewer true detections.

    Returns:
        (H, W) bool mask: True where the cell exceeds the local CFAR threshold.
    """
    if power.ndim != 2:
        raise ValueError(f"power must be 2D, got shape {power.shape}")
    if (power < 0).any():
        raise ValueError("power must be non-negative")

    H, W = power.shape
    tr_h, tr_w = train_size
    g_h, g_w = guard_size

    # Build an integral image so we can compute the sum over any rectangle in
    # O(1). The standard "pad with zero row/col on top/left" lets us treat
    # negative clamped indices uniformly.
    integ = np.pad(np.cumsum(np.cumsum(power.astype(np.float64), axis=0), axis=1), ((1, 0), (1, 0)))

    def rect_sum_and_count(r0: np.ndarray, r1: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r0c = np.clip(r0, 0, H)
        r1c = np.clip(r1, 0, H)
        c0c = np.clip(c0, 0, W)
        c1c = np.clip(c1, 0, W)
        s = integ[r1c, c1c] - integ[r0c, c1c] - integ[r1c, c0c] + integ[r0c, c0c]
        cnt = np.maximum((r1c - r0c) * (c1c - c0c), 1)
        return s, cnt

    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    outer_sum, outer_cnt = rect_sum_and_count(
        rr - tr_h - g_h,
        rr + tr_h + g_h + 1,
        cc - tr_w - g_w,
        cc + tr_w + g_w + 1,
    )
    inner_sum, inner_cnt = rect_sum_and_count(
        rr - g_h,
        rr + g_h + 1,
        cc - g_w,
        cc + g_w + 1,
    )
    train_sum = outer_sum - inner_sum
    train_cnt = np.maximum(outer_cnt - inner_cnt, 1)

    # The per-cell α depends on the actual training count (which may shrink
    # at the array edges). For consistency we use the nominal count.
    n_train_nominal = (2 * (tr_h + g_h) + 1) * (2 * (tr_w + g_w) + 1) - (2 * g_h + 1) * (2 * g_w + 1)
    if n_train_nominal < 1:
        raise ValueError(f"train_size {train_size} ≤ guard_size {guard_size}; no training cells")
    alpha = n_train_nominal * (pfa ** (-1.0 / n_train_nominal) - 1.0)

    threshold = (train_sum / train_cnt) * alpha
    return power > threshold


def cube_to_detections(
    cube: np.ndarray,
    config: RadarConfig,
    pfa: float = 1e-3,
    min_range_bin: int = 4,
    max_range_bin: int | None = None,
    train_size: tuple[int, int] = (8, 4),
    guard_size: tuple[int, int] = (4, 2),
    elevation_bin_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Run CFAR on the (range × azimuth) plane of max-doppler-and-elevation
    power, then for each detection record the argmax Doppler and Elevation
    bins. Returns an (N, 5) array of detections in BIN coordinates:

        columns = (range_bin, azimuth_bin, elevation_bin, doppler_bin, peak_power)

    `elevation_bin_range` clamps the argmax-elevation search to a contiguous
    sub-range [lo, hi). The AWR1843 has only 2 physical elevation MIMO rows
    that are zero-padded to 8 bins; the outer elevation bins are essentially
    hallucinated by the FFT and cause z-axis spread far outside the scene's
    real vertical extent (verified empirically on B408 indoor — LIRA z spread
    8 m vs LiDAR 3 m). Default (3, 5) keeps the central 2 bins (≈ ±10°),
    matching the radar's actual elevation resolution. Pass (0, El) to recover
    the unrestricted behavior.
    """
    if cube.ndim != 4:
        raise ValueError(f"cube must be 4D (R, D, Az, El), got {cube.shape}")
    R, D, Az, El = cube.shape

    if max_range_bin is None:
        max_range_bin = R
    max_range_bin = min(int(max_range_bin), R)
    min_range_bin = max(int(min_range_bin), 0)
    if min_range_bin >= max_range_bin:
        raise ValueError(f"min_range_bin {min_range_bin} ≥ max_range_bin {max_range_bin}")

    if elevation_bin_range is None:
        el_lo, el_hi = max(0, El // 2 - 1), min(El, El // 2 + 1)
    else:
        el_lo, el_hi = elevation_bin_range
    if not (0 <= el_lo < el_hi <= El):
        raise ValueError(
            f"elevation_bin_range {(el_lo, el_hi)} must satisfy 0 ≤ lo < hi ≤ {El}"
        )

    # Power summary: max over Doppler, then argmax-elevation within the
    # physically reasonable subrange.
    power_rda = cube.max(axis=1)  # (R, Az, El)
    el_slice = power_rda[..., el_lo:el_hi]  # (R, Az, el_hi-el_lo)
    el_argmax_local = el_slice.argmax(axis=-1)  # (R, Az), values in [0, el_hi-el_lo)
    el_argmax = el_argmax_local + el_lo
    power_ra = np.take_along_axis(power_rda, el_argmax[..., None], axis=-1)[..., 0]

    mask = ca_cfar_2d(power_ra, train_size=train_size, guard_size=guard_size, pfa=pfa)
    mask[:min_range_bin, :] = False
    mask[max_range_bin:, :] = False

    if not mask.any():
        return np.zeros((0, 5), dtype=np.float64)

    r_idx, a_idx = np.where(mask)
    el_idx = el_argmax[r_idx, a_idx]
    dop_slice = cube[r_idx, :, a_idx, el_idx]  # (N, D)
    d_idx = dop_slice.argmax(axis=1)
    peaks = power_ra[r_idx, a_idx]

    return np.stack(
        [r_idx.astype(np.float64), a_idx.astype(np.float64), el_idx.astype(np.float64),
         d_idx.astype(np.float64), peaks.astype(np.float64)],
        axis=1,
    )


def detections_to_points(
    detections: np.ndarray,
    config: RadarConfig,
) -> np.ndarray:
    """Convert (N, 5) bin-coord detections into radar-local Cartesian points.

    Output: (N, 5) float64 array with columns
        (x, y, z, intensity, doppler_m_s)
    in the radar's local frame following the RadarEyes Z_UP convention:
        +X = right,  +Y = forward (look-axis),  +Z = up.
    """
    if detections.size == 0:
        return np.zeros((0, 5), dtype=np.float64)

    geom = build_bin_geometry(config)
    range_m = geom["range_m"]
    doppler_m_s = geom["doppler_m_s"]
    azimuth_rad = geom["azimuth_rad"]
    elevation_rad = geom["elevation_rad"]

    r_bin = detections[:, 0].astype(np.int64)
    a_bin = detections[:, 1].astype(np.int64)
    e_bin = detections[:, 2].astype(np.int64)
    d_bin = detections[:, 3].astype(np.int64)
    peak = detections[:, 4]

    r_m = range_m[r_bin]
    az = azimuth_rad[a_bin]
    el = elevation_rad[e_bin]
    dop = doppler_m_s[d_bin]

    x = r_m * np.sin(az) * np.cos(el)
    y = r_m * np.cos(az) * np.cos(el)   # forward
    z = r_m * np.sin(el)
    return np.stack([x, y, z, peak, dop], axis=1)
