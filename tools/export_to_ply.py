"""Export a trained RadarSplat checkpoint to standard 3DGS PLY format.

The output PLY can be opened directly in any 3D Gaussian Splat viewer:
  * SuperSplat (web)  : https://playcanvas.com/supersplat/editor   (drag-drop)
  * antimatter15/splat-viewer (web)
  * gsplat.js (web)
  * Official Inria viewer
  * Nerfstudio's `ns-viewer`
  * UnityGaussianSplatting plugin (Unity Editor)

The format matches the layout used by the official 3DGS code, so all
viewers should accept it without modification.

Usage:
    python tools/export_to_ply.py \\
        --run-dir outputs/rsv1_alpha0.01_B408 \\
        --out outputs/rsv1_alpha0.01_B408/gaussians.ply
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.baselines._gsplat_common import SH_COEFS_REST, load_checkpoint  # noqa: E402


def _ply_header(n_points: int) -> bytes:
    """Standard 3DGS PLY header: position, normal, sh0 (3), shN (rest 45 for SH3),
    opacity, scale (3), rot (4) — all float32 little-endian, vertex element."""
    # field count: 3 (xyz) + 3 (nxyz) + 3 (sh0) + (SH_COEFS_REST*3 = 45) + 1 (op) + 3 (scale) + 4 (rot) = 62
    header = []
    header.append("ply")
    header.append("format binary_little_endian 1.0")
    header.append(f"element vertex {n_points}")
    header.append("property float x")
    header.append("property float y")
    header.append("property float z")
    header.append("property float nx")
    header.append("property float ny")
    header.append("property float nz")
    for i in range(3):  # sh0 DC components
        header.append(f"property float f_dc_{i}")
    for i in range(SH_COEFS_REST * 3):  # shN flattened: 15 coefs × 3 channels = 45
        header.append(f"property float f_rest_{i}")
    header.append("property float opacity")
    for i in range(3):
        header.append(f"property float scale_{i}")
    for i in range(4):
        header.append(f"property float rot_{i}")
    header.append("end_header")
    return ("\n".join(header) + "\n").encode("ascii")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default=None,
                   help="Run dir containing checkpoint/checkpoints/")
    p.add_argument("--checkpoint", default=None,
                   help="Direct path to a .pt; overrides --run-dir")
    p.add_argument("--out", default=None,
                   help="Output .ply path; defaults to <run>/gaussians.ply")
    args = p.parse_args()

    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.run_dir:
        ckpt_dir = os.path.join(args.run_dir, "checkpoint", "checkpoints")
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        if not ckpts:
            print(f"error: no .pt checkpoint in {ckpt_dir}", file=sys.stderr)
            return 2
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    else:
        print("error: provide --run-dir or --checkpoint", file=sys.stderr)
        return 2
    print(f"loading {ckpt_path}")
    gm = load_checkpoint(ckpt_path, device="cpu")
    n = gm.num_gaussians
    print(f"  {n} Gaussians")

    means = gm.means.detach().cpu().numpy().astype(np.float32)   # (N, 3)
    scales = gm.scales.detach().cpu().numpy().astype(np.float32)  # (N, 3) log-space
    quats = gm.quats.detach().cpu().numpy().astype(np.float32)    # (N, 4) wxyz, may be unnormalized
    # Normalize quats for downstream viewers.
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True).clip(min=1e-8)
    opacities = gm.opacities.detach().cpu().numpy().astype(np.float32)  # (N,) logit
    sh0 = gm.sh0.detach().cpu().numpy().astype(np.float32)[:, 0, :]     # (N, 3) DC
    # shN: official 3DGS stores SHs in (n_coeffs, 3) per-Gaussian and then flattens
    # column-major-ish. The convention that the Inria viewer expects is the
    # 45 floats ordered by (coef_idx * 3 + channel_idx) — i.e. coef0_r, coef0_g,
    # coef0_b, coef1_r, ... which is the channel-last flatten of (N, K, 3).
    # Some viewers expect channel-first. We do channel-LAST, the Inria
    # convention; SuperSplat understands both.
    shN = gm.shN.detach().cpu().numpy().astype(np.float32)              # (N, 15, 3)
    shN_flat = shN.reshape(n, -1)                                       # (N, 45)

    normals = np.zeros((n, 3), dtype=np.float32)
    # Per-Gaussian record: 3 + 3 + 3 + 45 + 1 + 3 + 4 = 62 floats.
    record = np.concatenate([
        means, normals, sh0, shN_flat,
        opacities.reshape(-1, 1), scales, quats
    ], axis=1).astype(np.float32, copy=False)
    assert record.shape[1] == 3 + 3 + 3 + 45 + 1 + 3 + 4, record.shape

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(args.run_dir, "gaussians.ply")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(_ply_header(n))
        f.write(record.tobytes(order="C"))
    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"wrote {out_path}  ({sz_mb:.1f} MB)")
    print()
    print("Drop this PLY into a 3DGS viewer to rotate it freely:")
    print("  - SuperSplat (web, drag-drop):  https://playcanvas.com/supersplat/editor")
    print("  - antimatter15 splat viewer:    https://antimatter15.com/splat/")
    print("  - or any Gaussian Splat viewer that accepts 3DGS-format PLY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
