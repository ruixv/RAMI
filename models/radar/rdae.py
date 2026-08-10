"""ADC → Range-Doppler-Azimuth-Elevation cube for TI AWR1843.

Implements the standard four-stage FFT chain for a TI AWR1843 radar with 3 Tx
× 4 Rx antennas operating in BPM TDM-MIMO mode:

    ADC complex (samples, chirps, rx=4, tx=3)
        ↓ range FFT (axis 0)
        ↓ doppler FFT + fftshift (axis 1)
        ↓ MIMO virtual-array unfold (rx, tx → 8 azimuth × 2 elevation)
        ↓ azimuth FFT + fftshift (axis 2, length 8)
        ↓ elevation FFT + fftshift (axis 3, length 8)
        ↓ magnitude
    cube float32 (range=128, doppler=N_chirps, azimuth=8, elevation=8)

The MIMO virtual-array layout is derived from the AWR1843's antenna placement
in BPM mode and matches the unfold used in
`SelfMadePackage/IPLab_mmwavePCD/radarPcdProcessing/` and
`SemanticRadar/radar_clip_alignment/radar_vl/data/radar_io.py`. Per the
project boundary rule (own repo, no imports across sibling projects), this
file is a clean reimplementation rather than a copy.

Physical constants (sample rate, ramp slope, frequencies) are sourced from
`SelfMadePackage/IPLab_mmwavePCD/config/1843_coloradar.yml` (azi) and
`1843_coherentEle.yml` (ele). They are correct for the RadarEyes capture
campaign and would only need re-derivation if Stereolabs replaced the radar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


C_LIGHT_M_S = 2.99792458e8


@dataclass(frozen=True)
class RadarConfig:
    """All physical and FFT parameters for one of the two 1843 arms."""

    name: str                     # "azi" or "ele"
    num_adc_samples: int          # always 128 in RadarEyes
    num_chirps_per_frame: int     # 128 (azi) or 8 (ele)
    num_rx: int                   # 4
    num_tx: int                   # 3
    start_freq_hz: float          # carrier start frequency
    chirp_slope_hz_per_s: float   # Kr
    chirp_ramp_time_s: float      # time the chirp ramps
    adc_start_time_s: float       # delay before ADC sampling within ramp
    idle_time_s: float            # idle between chirps
    adc_sample_rate_hz: float     # Fs

    # Output cube shape (post FFTs / MIMO unfold). For 1843 these are 8 × 8.
    n_azimuth_bins: int = 8
    n_elevation_bins: int = 8

    # MIMO virtual-array unfold table. Default matches 1843 BPM layout used by
    # the legacy pipeline. Each row is (start_az, end_az, tx_index, el_index)
    # meaning "place rx 0..3 of the given tx into mimo[start_az:end_az, el]".
    mimo_unfold: tuple[tuple[int, int, int, int], ...] = field(
        default_factory=lambda: (
            (0, 4, 0, 0),   # Tx 0 → az 0-3, el 0
            (4, 8, 2, 0),   # Tx 2 → az 4-7, el 0
            (2, 6, 1, 1),   # Tx 1 → az 2-5, el 1
        )
    )

    # --- derived ---
    @property
    def chirp_period_s(self) -> float:
        """Time between consecutive Doppler samples (per logical chirp)."""
        return self.idle_time_s + self.chirp_ramp_time_s + self.adc_start_time_s

    @property
    def chirp_bandwidth_hz(self) -> float:
        """Bandwidth used by the ADC during one chirp ramp."""
        return self.chirp_slope_hz_per_s * (self.num_adc_samples / self.adc_sample_rate_hz)

    @property
    def range_resolution_m(self) -> float:
        return C_LIGHT_M_S / (2.0 * self.chirp_bandwidth_hz)

    @property
    def wavelength_m(self) -> float:
        """Wavelength at the center of the chirp."""
        f_center = self.start_freq_hz + 0.5 * self.chirp_bandwidth_hz
        return C_LIGHT_M_S / f_center

    @property
    def max_velocity_m_s(self) -> float:
        """Maximum unambiguous Doppler velocity."""
        return self.wavelength_m / (4.0 * self.chirp_period_s)

    @property
    def doppler_resolution_m_s(self) -> float:
        """Per-bin Doppler velocity step after fftshift, spans [-max, +max]."""
        return (2.0 * self.max_velocity_m_s) / self.num_chirps_per_frame


# Locked configs derived from
# SelfMadePackage/IPLab_mmwavePCD/config/1843_coloradar.yml and
# 1843_coherentEle.yml.

AZI_CONFIG = RadarConfig(
    name="azi",
    num_adc_samples=128,
    num_chirps_per_frame=128,
    num_rx=4,
    num_tx=3,
    start_freq_hz=77.0e9,
    chirp_slope_hz_per_s=1.00000000377e14,  # ≈ 100 MHz / μs
    chirp_ramp_time_s=12.0e-6,
    adc_start_time_s=7.0e-6,
    idle_time_s=110.0e-6,
    adc_sample_rate_hz=10.666e6,
)

ELE_CONFIG = RadarConfig(
    name="ele",
    num_adc_samples=128,
    num_chirps_per_frame=8,
    num_rx=4,
    num_tx=3,
    start_freq_hz=79.0e9,
    chirp_slope_hz_per_s=5.0e13,
    chirp_ramp_time_s=128.0 / 4.0e6,  # numAdcSamples / Fs
    adc_start_time_s=5.0e-6,
    idle_time_s=30.0e-6,
    adc_sample_rate_hz=4.0e6,
)


def adc_to_rdae_cube(adc: np.ndarray, config: RadarConfig) -> np.ndarray:
    """Run the four-stage FFT chain and return |cube| as float32.

    Args:
        adc: complex array (samples, chirps, rx, tx). Shape must match the
             config's declared dims (mostly so we catch the azi-vs-ele mixup
             at the call site rather than silently produce a bogus cube).
        config: RadarConfig with the FFT sizes and MIMO unfold table.

    Returns:
        Magnitude cube of shape
        (num_adc_samples, num_chirps_per_frame, n_azimuth_bins, n_elevation_bins).
    """
    expected = (config.num_adc_samples, config.num_chirps_per_frame, config.num_rx, config.num_tx)
    if adc.shape != expected:
        raise ValueError(f"adc shape {adc.shape} does not match {config.name} config {expected}")
    if not np.iscomplexobj(adc):
        raise TypeError(f"adc must be complex, got {adc.dtype}")

    n_samples = config.num_adc_samples
    n_chirps = config.num_chirps_per_frame
    n_az = config.n_azimuth_bins
    n_el = config.n_elevation_bins

    rng = np.fft.fft(adc, n_samples, axis=0)
    dop = np.fft.fftshift(np.fft.fft(rng, n_chirps, axis=1), axes=1)

    # MIMO virtual-array unfold: (rx=4, tx=3) → (azimuth=8, elevation=2),
    # then we zero-pad the elevation dim to n_el for the elevation FFT.
    mimo = np.zeros((n_samples, n_chirps, n_az, 2), dtype=dop.dtype)
    for start_az, end_az, tx_idx, el_idx in config.mimo_unfold:
        if (end_az - start_az) != config.num_rx:
            raise ValueError(
                f"mimo_unfold row {(start_az, end_az, tx_idx, el_idx)} expects "
                f"{config.num_rx} azimuth slots but spans {end_az - start_az}"
            )
        mimo[:, :, start_az:end_az, el_idx] = dop[:, :, :, tx_idx]

    az = np.fft.fftshift(np.fft.fft(mimo, n_az, axis=2), axes=2)
    elv = np.fft.fftshift(np.fft.fft(az, n_el, axis=3), axes=3)
    return np.abs(elv).astype(np.float32)


def build_bin_geometry(config: RadarConfig) -> dict:
    """Return per-axis physical bin coordinates of the RDAE cube.

    Keys:
        range_m       — (num_adc_samples,) float64, bin centers in meters
        doppler_m_s   — (num_chirps_per_frame,) float64, signed velocity
        azimuth_rad   — (n_azimuth_bins,) float64
        elevation_rad — (n_elevation_bins,) float64
        range_res_m   — scalar
        doppler_res_m_s — scalar
    """
    n_chirps = config.num_chirps_per_frame
    range_m = np.arange(config.num_adc_samples, dtype=np.float64) * config.range_resolution_m
    doppler_m_s = (np.arange(n_chirps, dtype=np.float64) - n_chirps // 2) * config.doppler_resolution_m_s

    # FFT-shifted angle bins: even spacing in sin-angle space across [-1, 1],
    # then arcsin to physical angles. Matches the convention in
    # SemanticRadar/radar_clip_alignment/radar_vl/data/radar_io.py.
    n_az = config.n_azimuth_bins
    n_el = config.n_elevation_bins
    az_sin = np.linspace(-1.0, 1.0, n_az, endpoint=False) + 1.0 / n_az
    el_sin = np.linspace(-1.0, 1.0, n_el, endpoint=False) + 1.0 / n_el
    azimuth_rad = np.arcsin(az_sin)
    elevation_rad = np.arcsin(el_sin)

    return {
        "range_m": range_m,
        "doppler_m_s": doppler_m_s,
        "azimuth_rad": azimuth_rad,
        "elevation_rad": elevation_rad,
        "range_res_m": float(config.range_resolution_m),
        "doppler_res_m_s": float(config.doppler_resolution_m_s),
    }
