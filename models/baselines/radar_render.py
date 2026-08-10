"""Differentiable radar forward rendering loss for 3D Gaussian Splatting.

Inspired by Radar Fields (Borts et al., SIGGRAPH 2024): instead of converting
radar to a hand-crafted occupancy grid (our prior free-space carving, which
failed), we RENDER the expected radar range-azimuth power map from the Gaussian
field via a physics-informed forward model (radar equation, 1/R^n falloff) and
supervise it against the MEASURED radar RA map. This is a LIGHT-INDEPENDENT
geometric constraint: it shapes Gaussian geometry where the photometric loss is
useless (dark regions), which is exactly where radar should earn its keep.

Central hypothesis under test: radar's contribution to NVS scales with darkness.

Coordinate conventions (must match the rest of the codebase exactly):
  - Radar-local frame (cfar.py:detections_to_points, RadarEyes Z_UP):
        +X = right, +Y = forward (boresight), +Z = up
        x = r sin(az) cos(el),  y = r cos(az) cos(el),  z = r sin(el)
  - world -> radar: xyz_radar = (xyz_world - O) @ R   (transforms.world_to_sensor)
        R = quat_to_R(body_quat);  O = body_pos + R @ SENSOR_OFFSETS["radar_azi"]
  - RA-map binning (rdae.build_bin_geometry):
        range bin  rb = r / range_res_m            (range_m[k] = k * range_res)
        azimuth    az_sin[b] = -1 + (2b+1)/n_az    => b = (sin_az + 1)*n_az/2 - 0.5
        measured RA from _ramap_from_adc: log1p(|cube|.max(doppler,el)), standardized.
"""
from __future__ import annotations
import numpy as np
import torch

from data.radareyes.sync import interpolate_pose
from data.radareyes.transforms import SENSOR_OFFSETS, quat_to_R
from models.radar.lira import AZI_CONFIG
from models.radar.rdae import build_bin_geometry
from tools.radar_depth_completion import _ramap_from_adc

_EPS = 1e-8


def build_radar_frames(scene, n_frames: int, device, stride_pad: int = 2):
    """Sample n_frames radar_azi frames across the trajectory; precompute for each
    the measured RA map and the world->radar pose (O, R). Returns a dict of torch
    tensors (built once at train start, reused every iteration).
    """
    stream = scene.radar_azi
    if stream is None:
        raise ValueError("scene has no radar_azi stream")
    N = len(stream.timestamps)
    lo, hi = stride_pad, max(stride_pad, N - 1 - stride_pad)
    idxs = np.unique(np.linspace(lo, hi, n_frames).astype(int))

    geom = build_bin_geometry(AZI_CONFIG)
    range_res = float(geom["range_res_m"])
    n_az = int(AZI_CONFIG.n_azimuth_bins)
    n_rng = int(AZI_CONFIG.num_adc_samples)

    Os, Rs, maps = [], [], []
    for idx in idxs:
        adc = stream.load_adc(int(idx))
        ra = _ramap_from_adc(adc)  # (n_rng, n_az) float32, log1p + standardized
        t_r = float(stream.timestamps[int(idx)])
        pp, pq = interpolate_pose(scene.zed.pose_stream, t_r)
        R = quat_to_R(pq)
        O = np.asarray(pp, dtype=np.float64) + R @ SENSOR_OFFSETS["radar_azi"]
        Os.append(O); Rs.append(R); maps.append(ra)

    return {
        "O": torch.tensor(np.array(Os), dtype=torch.float32, device=device),      # (F,3)
        "R": torch.tensor(np.array(Rs), dtype=torch.float32, device=device),      # (F,3,3)
        "ra": torch.tensor(np.array(maps), dtype=torch.float32, device=device),   # (F,n_rng,n_az)
        "range_res": range_res, "n_az": n_az, "n_rng": n_rng, "n_frames": len(idxs),
    }


def _standardize(x):
    return (x - x.mean()) / (x.std() + _EPS)


def render_ra_map(means, opacities, O, R, range_res, n_rng, n_az,
                  refl=None, power_exp: float = 4.0, az_sin_max: float = 0.95):
    """Differentiable forward render of one radar RA power map from the Gaussians.

    means      (Ng,3) world-frame Gaussian centers (live params)
    opacities  (Ng,)  in [0,1]
    O (3,), R (3,3)   world->radar pose for this frame
    refl       (Ng,) optional per-Gaussian radar reflectance in [0,1]; default 1
    returns    (n_rng, n_az) predicted power map (NOT yet standardized)

    Gradient flows to `means` through both the bilinear splat weights (rb, ab)
    and the per-Gaussian power; to `opacities`/`refl` through power.
    """
    p = (means - O[None, :]) @ R                      # (Ng,3) radar frame
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = torch.sqrt(x * x + y * y + z * z + _EPS)
    horiz = torch.sqrt(x * x + y * y + _EPS)
    sin_az = x / horiz                                # in [-1,1]

    rb = r / range_res                                # continuous range bin
    ab = (sin_az + 1.0) * (n_az * 0.5) - 0.5          # continuous azimuth bin

    # FOV / validity: in front (y>0), within range window, within azimuth cone.
    valid = (y > 0) & (rb >= 0) & (rb <= n_rng - 1) & (sin_az.abs() < az_sin_max)
    power = opacities / (r ** power_exp + _EPS)
    if refl is not None:
        power = power * refl
    power = power * valid.to(power.dtype)

    # bilinear splat into (n_rng, n_az) — differentiable via continuous weights.
    rb_c = rb.clamp(0, n_rng - 1)
    ab_c = ab.clamp(0, n_az - 1)
    r0 = torch.floor(rb_c).long(); a0 = torch.floor(ab_c).long()
    r1 = (r0 + 1).clamp(max=n_rng - 1); a1 = (a0 + 1).clamp(max=n_az - 1)
    wr1 = rb_c - r0.to(rb_c.dtype); wr0 = 1.0 - wr1
    wa1 = ab_c - a0.to(ab_c.dtype); wa0 = 1.0 - wa1

    img = means.new_zeros(n_rng * n_az)
    for (ri, ai, wi) in ((r0, a0, wr0 * wa0), (r0, a1, wr0 * wa1),
                         (r1, a0, wr1 * wa0), (r1, a1, wr1 * wa1)):
        idx = ri * n_az + ai
        img = img.index_add(0, idx, power * wi)
    return img.view(n_rng, n_az)


def radar_render_loss(means, opacities, frames, k_frames, rng,
                      refl=None, power_exp: float = 2.0):
    """Mean MSE between standardized predicted and measured RA maps over a random
    subset of k_frames radar frames. Light-independent geometric supervision."""
    F = frames["n_frames"]
    k = min(k_frames, F)
    pick = rng.choice(F, size=k, replace=False)
    n_rng, n_az, range_res = frames["n_rng"], frames["n_az"], frames["range_res"]
    losses = []
    for j in pick:
        j = int(j)
        pred = render_ra_map(means, opacities, frames["O"][j], frames["R"][j],
                             range_res, n_rng, n_az, refl=refl, power_exp=power_exp)
        pred = _standardize(torch.log1p(pred))
        losses.append(((pred - frames["ra"][j]) ** 2).mean())
    return torch.stack(losses).mean()
