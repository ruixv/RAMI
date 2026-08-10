"""Stage-1b probe: RADAR AS ARBITER between two geometry hypotheses.

Probe #1 (radar_referee_probe.py) showed the ABSOLUTE radar health score is
confounded by Gaussian count (diffuse mass inflates Pearson with the noisy
measured RA map). Fix: a RELATIVE judgment on EQUAL support — for the same
pixels of the same view, unproject (a) the depth-PRIOR and (b) the current
MODEL's rendered depth into two equal-size world point sets, forward-render
both into radar RA maps, and ask which agrees better with the MEASURED map.
Equal point count + equal support kills the mass confound by construction.

If the arbiter's per-scene preference matches the KNOWN 3DGS outcome of
adding that prior (helps/hurts, 7 cases incl. dark401a which flips sign
between weak and strong baselines), then radar can drive an adaptive
prior-gating controller during training: the first viable radar mechanism
after all per-pixel supervision uses failed.

Cases (ckpt = baseline WITHOUT prior; outcome = PSNR(prior arm) - PSNR(ckpt arm)):
  anchored DA-V2 priors (lambda=0.5):
    dark401a light rgbonly 22.61 + radarglobal prior -> 20.13   HURT  (-2.48)
    n0703c   light rgbonly 17.54 + radarglobal prior -> 19.20   HELP  (+1.66)
    dark401a heavy rgbonly 18.10 + radarglobal prior -> 20.99   HELP  (+2.89)
    n0703c   heavy rgbonly 16.89 + radarglobal prior -> 18.69   HELP  (+1.80)
  learned RadarDepthNet priors (lambda=0.1):
    n0703a noprior 23.25 + rgb -> 23.02 HURT(-0.23) | + radar -> 22.02 HURT(-1.23)
    n0630c noprior 24.67 + rgb -> 25.58 HELP(+0.91) | + radar -> 25.30 HELP(+0.63)
    n0703c noprior 17.87 + rgb -> 19.28 HELP(+1.41) | + radar -> 19.07 HELP(+1.20)

Usage: python tools/radar_arbiter_probe.py [--n-views 15] [--n-px 20000]
Output: outputs/referee_probe/ARBITER.md
"""
from __future__ import annotations
import argparse, glob, json, os, sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from data.radareyes.sync import interpolate_pose
from data.radareyes.transforms import SENSOR_OFFSETS, quat_to_R
from models.baselines.dn_splatter import _resolve_scene_from_splits
from models.baselines.radar_render import render_ra_map, _ramap_from_adc
from models.radar.lira import AZI_CONFIG
from models.radar.rdae import build_bin_geometry
from tools.fov_mask import project_to_camera
from tools.zed_calibration import ZED_BODY_TO_OPTICAL_R

CASES = [
    # name, tag, ckpt_run, prior_dir, known_delta_psnr
    ("dark401a_light_anch", "dark401a",
     "outputs/anchored_depth_light/dark401a_rgbonly",
     "outputs/anchored_depth/dark401a_radarglobal", -2.48),
    ("n0703c_light_anch", "n0703c",
     "outputs/anchored_depth_light/n0703c_rgbonly",
     "outputs/anchored_depth/n0703c_radarglobal", +1.66),
    ("dark401a_heavy_anch", "dark401a",
     "outputs/anchored_depth_gonogo/dark401a_rgbonly",
     "outputs/anchored_depth/dark401a_radarglobal", +2.89),
    ("n0703c_heavy_anch", "n0703c",
     "outputs/anchored_depth_gonogo/n0703c_rgbonly",
     "outputs/anchored_depth/n0703c_radarglobal", +1.80),
    ("n0703a_net_rgb", "n0703a",
     "outputs/depthprior_ablation/n0703a_noprior",
     "outputs/depth_priors/n0703a/rgb", -0.23),
    ("n0703a_net_radar", "n0703a",
     "outputs/depthprior_ablation/n0703a_noprior",
     "outputs/depth_priors/n0703a/radar", -1.23),
    ("n0630c_net_rgb", "n0630c",
     "outputs/depthprior_ablation/n0630c_noprior",
     "outputs/depth_priors/n0630c/rgb", +0.91),
    ("n0630c_net_radar", "n0630c",
     "outputs/depthprior_ablation/n0630c_noprior",
     "outputs/depth_priors/n0630c/radar", +0.63),
    ("n0703c_net_rgb", "n0703c",
     "outputs/depthprior_ablation/n0703c_noprior",
     "outputs/depth_priors/n0703c/rgb", +1.41),
    ("n0703c_net_radar", "n0703c",
     "outputs/depthprior_ablation/n0703c_noprior",
     "outputs/depth_priors/n0703c/radar", +1.20),
]


def _pearson(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def build_radar_frames_t(scene, n_frames, device, stride_pad=2):
    """Like radar_render.build_radar_frames but also keeps per-frame timestamps."""
    stream = scene.radar_azi
    N = len(stream.timestamps)
    lo, hi = stride_pad, max(stride_pad, N - 1 - stride_pad)
    idxs = np.unique(np.linspace(lo, hi, n_frames).astype(int))
    geom = build_bin_geometry(AZI_CONFIG)
    Os, Rs, maps, ts = [], [], [], []
    for idx in idxs:
        adc = stream.load_adc(int(idx))
        ra = _ramap_from_adc(adc)
        t_r = float(stream.timestamps[int(idx)])
        pp, pq = interpolate_pose(scene.zed.pose_stream, t_r)
        R = quat_to_R(pq)
        O = np.asarray(pp, dtype=np.float64) + R @ SENSOR_OFFSETS["radar_azi"]
        Os.append(O); Rs.append(R); maps.append(ra); ts.append(t_r)
    return {
        "O": torch.tensor(np.array(Os), dtype=torch.float32, device=device),
        "R": torch.tensor(np.array(Rs), dtype=torch.float32, device=device),
        "ra": torch.tensor(np.array(maps), dtype=torch.float32, device=device),
        "t": np.array(ts),
        "range_res": float(geom["range_res_m"]),
        "n_az": int(AZI_CONFIG.n_azimuth_bins),
        "n_rng": int(AZI_CONFIG.num_adc_samples),
        "n_frames": len(idxs),
    }


def _w2c_from_pose(zp, zq):
    """4x4 world->optical matching project_to_camera()'s convention exactly."""
    R = quat_to_R(zq)                       # body -> world
    Rw2o = ZED_BODY_TO_OPTICAL_R @ R.T      # world -> optical rotation
    w2c = np.eye(4)
    w2c[:3, :3] = Rw2o
    w2c[:3, 3] = -Rw2o @ np.asarray(zp, dtype=np.float64)
    return w2c


def _unproject(u, v, d, K, zp, zq):
    """pixels+optical-depth -> world; inverse of project_to_camera."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    pts_opt = np.stack([x, y, d], axis=-1)
    pts_body = pts_opt @ ZED_BODY_TO_OPTICAL_R       # inverse of @ R_b2o.T
    R = quat_to_R(zq)
    return pts_body @ R.T + np.asarray(zp, dtype=np.float64)


def _verify_roundtrip(K, zp, zq, W, H):
    rng = np.random.default_rng(0)
    u = rng.uniform(0, W - 1, 100); v = rng.uniform(0, H - 1, 100)
    d = rng.uniform(0.5, 15.0, 100)
    pw = _unproject(u, v, d, K, zp, zq)
    px, dp, va = project_to_camera(pw, zp, zq, fx=K[0, 0], fy=K[1, 1],
                                   cx=K[0, 2], cy=K[1, 2], image_w=W, image_h=H,
                                   near=0.05, far=50.0,
                                   body_to_optical_R=ZED_BODY_TO_OPTICAL_R)
    assert va.all(), "roundtrip: some points invalid"
    perr = np.abs(px - np.stack([u, v], -1)).max()
    derr = np.abs(dp - d).max()
    assert perr < 1e-3 and derr < 1e-3, f"roundtrip err px={perr} d={derr}"


@torch.no_grad()
def render_model_depth(ck, w2c, K, W, H, device):
    import gsplat
    means = ck["means"].to(device)
    ok = torch.isfinite(means).all(dim=1)
    means = means[ok]
    quats = torch.nn.functional.normalize(ck["quats"].to(device)[ok], dim=-1)
    scales = torch.exp(ck["scales"].to(device)[ok])
    opac = torch.sigmoid(ck["opacities"].to(device)[ok])
    sh0 = ck["sh0"].to(device)[ok]
    viewmat = torch.tensor(w2c, dtype=torch.float32, device=device).unsqueeze(0)
    Kt = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)
    renders, alphas, _ = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opac,
        colors=sh0, viewmats=viewmat, Ks=Kt, width=W, height=H,
        sh_degree=0, packed=False, render_mode="RGB+ED",
    )
    return renders[0, :, :, 3].cpu().numpy(), alphas[0, :, :, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-views", type=int, default=15)
    ap.add_argument("--n-px", type=int, default=20000)
    ap.add_argument("--n-radar-frames", type=int, default=160)
    ap.add_argument("--k-nearest", type=int, default=6)
    ap.add_argument("--power-exp", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/referee_probe")
    args = ap.parse_args()
    device = args.device
    rng = np.random.default_rng(0)

    scene_cache = {}
    results = []
    for name, tag, run_dir, prior_dir, known in CASES:
        cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint/checkpoints/gaussians_iter*.pt")))
        if not cks or not os.path.isdir(prior_dir):
            print(f"[{name}] SKIP (missing ckpt or prior dir)")
            continue
        if tag not in scene_cache:
            ed = f"outputs/canonical_gt/{tag}"
            splits = json.load(open(os.path.join(ed, "splits.json")))
            scene = _resolve_scene_from_splits(ed)
            frames = build_radar_frames_t(scene, args.n_radar_frames, device)
            intr = splits["intrinsics"]
            K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]],
                          [0, 0, 1]], np.float64)
            W, H = splits["image_size"]
            scene_cache[tag] = (splits, scene, frames, K, W, H)
            print(f"[{tag}] {frames['n_frames']} radar frames built")
        splits, scene, frames, K, W, H = scene_cache[tag]
        ck = torch.load(cks[-1], map_location="cpu", weights_only=True)

        recs = [r for r in splits["records"]
                if os.path.exists(os.path.join(prior_dir, r["filename"] + ".npz"))]
        sel = recs[:: max(1, len(recs) // args.n_views)][: args.n_views]
        deltas, prior_wins = [], 0
        pe_m_all, pe_p_all = [], []
        for ri, rec in enumerate(sel):
            t_img = float(rec["t_image"])
            zp, zq = interpolate_pose(scene.zed.pose_stream, t_img)
            if ri == 0:
                _verify_roundtrip(K, zp, zq, W, H)
            z = np.load(os.path.join(prior_dir, rec["filename"] + ".npz"))
            pdep = z["depth"].astype(np.float64)
            pconf = z["conf"].astype(np.float64) if "conf" in z else np.ones_like(pdep)
            w2c = _w2c_from_pose(zp, zq)
            mdep, malpha = render_model_depth(ck, w2c, K, W, H, device)
            ok = (np.isfinite(pdep) & (pdep > 0.3) & (pdep < 20.0) & (pconf > 0)
                  & np.isfinite(mdep) & (mdep > 0.3) & (mdep < 20.0) & (malpha > 0.5))
            ys, xs = np.nonzero(ok)
            if len(ys) < 1000:
                continue
            pick = rng.choice(len(ys), min(args.n_px, len(ys)), replace=False)
            u = xs[pick].astype(np.float64); v = ys[pick].astype(np.float64)
            pw_prior = _unproject(u, v, pdep[ys[pick], xs[pick]], K, zp, zq)
            pw_model = _unproject(u, v, mdep[ys[pick], xs[pick]], K, zp, zq)
            tp = torch.tensor(pw_prior, dtype=torch.float32, device=device)
            tm = torch.tensor(pw_model, dtype=torch.float32, device=device)
            ones = torch.ones(tp.shape[0], device=device)

            near = np.argsort(np.abs(frames["t"] - t_img))[: args.k_nearest]
            dpe = []
            for j in near:
                j = int(j)
                meas = frames["ra"][j].cpu().numpy()
                ras = []
                for pts in (tp, tm):
                    pred = render_ra_map(pts, ones, frames["O"][j], frames["R"][j],
                                         frames["range_res"], frames["n_rng"],
                                         frames["n_az"], power_exp=args.power_exp)
                    ras.append(torch.log1p(pred).cpu().numpy())
                if ras[0].sum() == 0 or ras[1].sum() == 0:
                    continue
                pe_p = _pearson(ras[0], meas); pe_m = _pearson(ras[1], meas)
                dpe.append(pe_p - pe_m); pe_p_all.append(pe_p); pe_m_all.append(pe_m)
            if dpe:
                dv = float(np.mean(dpe))
                deltas.append(dv)
                prior_wins += dv > 0
        if not deltas:
            print(f"[{name}] no usable views")
            continue
        md = float(np.mean(deltas))
        res = {"case": name, "tag": tag, "n_views": len(deltas),
               "delta_mean": md, "delta_med": float(np.median(deltas)),
               "frac_prior_wins": prior_wins / len(deltas),
               "pe_prior": float(np.mean(pe_p_all)), "pe_model": float(np.mean(pe_m_all)),
               "known_dpsnr": known,
               "sign_match": (md > 0) == (known > 0)}
        results.append(res)
        print(f"[{name}] arbiter Δpearson={md:+.4f} (med {res['delta_med']:+.4f}, "
              f"prior wins {res['frac_prior_wins']:.0%} of {len(deltas)} views; "
              f"pe_prior={res['pe_prior']:+.3f} pe_model={res['pe_model']:+.3f}) "
              f"known ΔPSNR={known:+.2f} -> {'MATCH' if res['sign_match'] else 'MISS'}")

    lines = ["# Radar-arbiter probe (prior-vs-model, equal support)", "",
             "| case | n_views | Δpearson(prior−model) | prior-win views | known ΔPSNR | sign match |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['case']} | {r['n_views']} | {r['delta_mean']:+.4f} | "
                     f"{r['frac_prior_wins']:.0%} | {r['known_dpsnr']:+.2f} | "
                     f"{'✅' if r['sign_match'] else '❌'} |")
    n_match = sum(r["sign_match"] for r in results)
    lines += ["", f"**Sign agreement: {n_match}/{len(results)}**"]
    # rank correlation between arbiter score and known ΔPSNR across cases
    if len(results) >= 3:
        xs = [r["delta_mean"] for r in results]; ys = [r["known_dpsnr"] for r in results]
        rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
        lines.append(f"Spearman(arbiter, ΔPSNR) across cases = "
                     f"{_pearson(rx.astype(float), ry.astype(float)):+.2f}")
    out = os.path.join(args.out, "ARBITER.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[arbiter] wrote {out}")


if __name__ == "__main__":
    main()
