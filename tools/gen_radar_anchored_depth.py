"""Radar-anchored foundation-model depth for LiDAR-free low-light 3DGS (2026-06-09).

Latest paradigm (RadarCam-Depth ICRA'24 / TacoDepth CVPR'25): a monocular DEPTH
FOUNDATION MODEL (Depth Anything V2) supplies dense relative structure; sparse
radar supplies METRIC SCALE via robust scale-shift alignment. This replaces our
old from-scratch tiny U-Net (which had poor structure -> radar made it worse).

Produces a per-view dense metric depth prior {depth, conf}.npz for radarsplat_v1.
Arms (isolate radar's contribution; NO LiDAR except the upper-bound arm):
  global : DA-V2 aligned to radar with ONE scene-global (s,t)         (no per-view radar)
  radar  : DA-V2 aligned per-view with robust radar scale-shift       (OURS)
  lidar  : LiDAR depth prior                                          (upper bound)

Also prints an intermediate GATE: per-view depth error vs LiDAR GT for global vs
radar alignment. If radar alignment isn't more accurate than global, the 3DGS
outcome is predictable -> skip the expensive runs.

Usage:
  python tools/gen_radar_anchored_depth.py --tag dark401a --mode radar --out <dir>
  python tools/gen_radar_anchored_depth.py --tag dark401a --gate   # just the depth eval
"""
from __future__ import annotations
import argparse, os, sys, json
import numpy as np
import torch
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from data.radareyes.sync import interpolate_pose, nearest_index
from data.radareyes.transforms import SENSOR_OFFSETS, sensor_to_world
from models.baselines.dn_splatter import _resolve_scene_from_splits, _build_lidar_depth_for_view
from models.baselines.radarsplat_physics import _build_lira_with_intensity
from models.radar.lira import AZI_CONFIG, adc_to_rdae_cube, cube_to_detections, detections_to_points
from models.baselines._gsplat_common import parse_colmap_export, colmap_image_to_w2c, K_from_camera
from tools.radar_depth_completion import _radar_maps
from tools.zed_calibration import ZED_BODY_TO_OPTICAL_R
from tools.fov_mask import project_to_camera

_EPS = 1e-6


def _sfm_depth_for_view(colmap, name, W, H):
    """No-radar metric anchor control: project COLMAP points3D into the view `name`
    -> sparse metric depth (u,v,z). Tests whether radar's metric scale beats the
    scene's own SfM scale (which RGB multi-view already provides)."""
    rng = np.zeros((H, W), np.float32); conf = np.zeros((H, W), np.float32)
    img = next((im for im in colmap.images.values() if im.name == name), None)
    if img is None or colmap.points3D.shape[0] == 0:
        return rng, conf
    cam = colmap.cameras[img.camera_id]
    K = K_from_camera(cam)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    sx, sy = W / float(cam.width), H / float(cam.height)
    w2c = colmap_image_to_w2c(img)
    X = colmap.points3D[:, :3]
    Xc = (w2c[:3, :3] @ X.T + w2c[:3, 3:4]).T          # (N,3) camera frame
    z = Xc[:, 2]
    m = z > 1e-3
    u = (fx * Xc[m, 0] / z[m] + cx) * sx
    v = (fy * Xc[m, 1] / z[m] + cy) * sy
    zz = z[m]
    ui = np.round(u).astype(int); vi = np.round(v).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    for uu, vv, dd in zip(ui[ok], vi[ok], zz[ok]):
        if rng[vv, uu] == 0 or dd < rng[vv, uu]:
            rng[vv, uu] = dd; conf[vv, uu] = 1.0
    return rng, conf


def _build_radar_depth_for_view(scene, t_image, K, W, H, device,
                                window_s=0.5, near=0.3, far=20.0, pfa=1e-2,
                                excl_idx=None):
    """Per-view radar depth: project CFAR detections from radar_azi frames within
    ±window_s of t_image to the camera view at t_image (temporally-local anchors,
    mirroring _build_lidar_depth_for_view). Returns (rng (H,W), conf (H,W)) with 0
    where no radar point projects. This replaces the single-t_tar LIRA snapshot
    (which starved views far from the snapshot time of anchors).
    excl_idx: optional set of radar-frame indices to skip entirely (strict-
    protocol controls: radar frames near test-view timestamps)."""
    rng = np.zeros((H, W), np.float32); conf = np.zeros((H, W), np.float32)
    stream = scene.radar_azi
    if stream is None:
        return rng, conf
    zp, zq = interpolate_pose(scene.zed.pose_stream, t_image)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    times = stream.timestamps
    i0 = max(0, int(nearest_index(times, t_image - window_s)))
    i1 = min(len(times) - 1, int(nearest_index(times, t_image + window_s)))
    for j in range(i0, i1 + 1):
        if excl_idx is not None and j in excl_idx:
            continue
        det = cube_to_detections(adc_to_rdae_cube(stream.load_adc(j), AZI_CONFIG), AZI_CONFIG,
                                 pfa=pfa, min_range_bin=4, max_range_bin=120)
        if det.shape[0] == 0:
            continue
        pts = np.asarray(detections_to_points(det, AZI_CONFIG))   # (n,5): x,y,z,intensity,doppler
        t_r = float(times[j])
        bp, bq = interpolate_pose(scene.zed.pose_stream, t_r)
        pw = sensor_to_world(pts[:, :3], bp, bq, SENSOR_OFFSETS["radar_azi"])[:, :3]
        wgt = pts[:, 3]
        px, dp, va = project_to_camera(pw.astype(np.float64), zp, zq, fx=fx, fy=fy, cx=cx, cy=cy,
                                       image_w=W, image_h=H, near=near, far=far,
                                       body_to_optical_R=ZED_BODY_TO_OPTICAL_R)
        if not va.any():
            continue
        u = px[va, 0].astype(int).clip(0, W - 1); v = px[va, 1].astype(int).clip(0, H - 1)
        d = dp[va].astype(np.float32); wv = wgt[va].astype(np.float32)
        for ui, vi, di, wi in zip(u, v, d, wv):
            for dv in (-1, 0, 1):           # 3x3 splat for radar's coarse angular footprint
                for du in (-1, 0, 1):
                    yy, xx = vi + dv, ui + du
                    if 0 <= yy < H and 0 <= xx < W and (rng[yy, xx] == 0 or di < rng[yy, xx]):
                        rng[yy, xx] = di; conf[yy, xx] = wi
    return rng, conf


_DA_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"   # overridable via --da-model


def _da_v2(device):
    from transformers import pipeline
    return pipeline(task="depth-estimation", model=_DA_MODEL, device=0)


def _rel_depth_from_da(pipe, pil_img, H, W):
    """DA-V2 disparity-like output -> relative DEPTH (higher disp = closer)."""
    out = pipe(pil_img)
    disp = np.asarray(out["depth"], np.float32)          # (h,w), ~[1,255], higher=closer
    disp = np.asarray(Image.fromarray(disp).resize((W, H), Image.BILINEAR), np.float32)
    disp = disp / 255.0
    rel = 1.0 / (disp + 1e-3)                            # relative depth (up to scale+shift)
    return rel


def _fit_scale_shift(rel, z_anchor, conf):
    """Weighted least squares z ≈ s*rel + t over anchor pixels. Returns (s,t)."""
    m = (conf > 0) & np.isfinite(z_anchor) & (z_anchor > 0)
    if m.sum() < 8:
        return None
    r = rel[m]; z = z_anchor[m]; w = conf[m]
    A = np.stack([r, np.ones_like(r)], 1)               # (n,2)
    W = w[:, None]
    AtA = (A * W).T @ A
    Atb = (A * W).T @ z
    try:
        s, t = np.linalg.solve(AtA + 1e-6 * np.eye(2), Atb)
    except np.linalg.LinAlgError:
        return None
    return float(s), float(t)


def _scene_views(tag, device, dav2=True):
    ed = f"outputs/canonical_gt/{tag}"
    splits = json.load(open(os.path.join(ed, "splits.json")))
    intr = splits["intrinsics"]; Wf, Hf = splits["image_size"]
    K = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
    scene = _resolve_scene_from_splits(ed)
    n2t = {r["filename"]: r["t_image"] for r in splits["records"]}
    t_tar = float(np.median(scene.radar_azi.timestamps))
    anc, w = _build_lira_with_intensity(scene, t_tar, {})
    Kt = torch.tensor(K, dtype=torch.float32, device=device)
    pipe = _da_v2(device) if dav2 else None
    return ed, splits, scene, n2t, anc, w, K, Kt, Wf, Hf, pipe


def _fit_subset(rel_list, z_list, cf_list, idx=None, k=None, seed=0):
    """Fit (s,b) on a concatenated anchor pool, optionally subsampled to k anchors.
    k==1 fixes b=0 (one range sample can only set scale)."""
    r = np.concatenate(rel_list); z = np.concatenate(z_list); c = np.concatenate(cf_list)
    if idx is not None:
        r, z, c = r[idx], z[idx], c[idx]
    if k is not None:
        rng = np.random.RandomState(seed)
        if k >= len(r):
            pass
        else:
            sel = rng.choice(len(r), size=k, replace=False)
            r, z, c = r[sel], z[sel], c[sel]
        if k == 1:
            return (float(z[0] / max(r[0], _EPS)), 0.0)
        if 2 <= len(r) < 8:   # below _fit_scale_shift's stability guard: solve directly
            A = np.stack([r, np.ones_like(r)], 1)
            Wm = c[:, None]
            try:
                s, b = np.linalg.solve((A * Wm).T @ A + 1e-6 * np.eye(2), (A * Wm).T @ z)
            except np.linalg.LinAlgError:
                return None
            return (float(s), float(b))
    if len(r) < 2:
        return None
    st = _fit_scale_shift(r, z, c)
    return st


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ed, splits, scene, n2t, anc, w, K, Kt, W, H, pipe = _scene_views(args.tag, device, dav2=(args.mode != "lidar"))
    n2split = {r["filename"]: r.get("split", "train") for r in splits["records"]}
    names = [r["filename"] for r in splits["records"] if r["filename"] in n2t]
    # Strict-protocol control: exclude radar frames near TEST-view timestamps
    # from anchor construction ("pf" = the single nearest frame per test time;
    # a float = all frames within +/- that many seconds of any test time).
    excl_idx = None
    if getattr(args, "exclude_test_radar", None):
        tr_times = np.asarray(scene.radar_azi.timestamps, float)
        test_ts = [float(n2t[nm]) for nm in n2t if n2split.get(nm) == "test"]
        excl_idx = set()
        if args.exclude_test_radar == "pf":
            for tt in test_ts:
                excl_idx.add(int(nearest_index(tr_times, tt)))
        else:
            wexc = float(args.exclude_test_radar)
            for tt in test_ts:
                excl_idx.update(np.where(np.abs(tr_times - tt) <= wexc)[0].tolist())
        print(f"[excl-test-radar] {args.tag}: {len(excl_idx)}/{len(tr_times)} radar "
              f"frames excluded (mode={args.exclude_test_radar})")
    if args.limit: names = names[:args.limit]
    if args.mode == "sfm":
        from tools.scene_registry import SCENES
        colmap = parse_colmap_export(SCENES[args.tag])
    else:
        colmap = None

    # Pass 1: gather per-view rel-depth + radar anchors (+ lidar for gate/global fit).
    skip_lidar = bool(args.fit_report or args.st_modes) and args.mode not in ("lidar", "lidarscale")
    per = []
    all_rel, all_zr, all_cf, all_tr = [], [], [], []   # all_tr: is-train flag per anchor block
    rel_meds_train, cam_pos_train = [], []
    for nm in names:
        t = float(n2t[nm])
        is_train = n2split.get(nm, "train") != "test"
        rgb = Image.open(os.path.join(ed, "images", nm)).convert("RGB")
        if args.alpha < 1.0:
            # Darkness-ladder honesty: DA-V2 must see the SAME darkness level as
            # training (no clean-image leakage). Same corrupt() as the training
            # dataset; deterministic per-view seed.
            from data.lowlight.noise_model import corrupt
            arr = corrupt(np.array(rgb, dtype=np.uint8), alpha=args.alpha,
                          seed=1 + abs(hash(nm)) % 100000)
            rgb = Image.fromarray(arr)
        rel = None if args.mode == "lidar" else _rel_depth_from_da(pipe, rgb, H, W)
        if args.mode == "sfm":
            rng, inten = _sfm_depth_for_view(colmap, nm, W, H)            # no-radar SfM metric anchors
        elif args.snapshot:
            rng, inten = _radar_maps(anc, w, t, scene, K, W, H, device)   # old single-t_tar snapshot
        else:
            rng, inten = _build_radar_depth_for_view(scene, t, K, W, H, device,
                                                     excl_idx=excl_idx)  # per-view temporally-local radar
        lid = (np.zeros((H, W), np.float32) if skip_lidar else
               _build_lidar_depth_for_view(scene, t, Kt, W, H, device=device).cpu().numpy().astype(np.float32))
        per.append((nm, rel, rng, inten, lid))
        if rel is not None and is_train:
            rel_meds_train.append(float(np.median(rel[::8, ::8])))
            zp, _zq = interpolate_pose(scene.zed.pose_stream, t)
            cam_pos_train.append(np.asarray(zp, np.float64))
        if rel is not None:
            m = (rng > 0)
            if m.any():
                all_rel.append(rel[m]); all_zr.append(rng[m]); all_cf.append(inten[m])
                all_tr.append(np.full(int(m.sum()), is_train, bool))

    # Global (s,t) from radar anchors across the scene (no per-view radar).
    # Always computed: needed by the 'global' arm AND by the gate comparison.
    # --fit-split train restricts the fit to anchors of training-split views.
    g_st = None
    tr_idx = (np.where(np.concatenate(all_tr))[0] if all_tr else None)
    if all_rel:
        if args.fit_split == "train" and tr_idx is not None and len(tr_idx) >= 8:
            g_st = _fit_subset(all_rel, all_zr, all_cf, idx=tr_idx)
        else:
            g_st = _fit_scale_shift(np.concatenate(all_rel), np.concatenate(all_zr), np.concatenate(all_cf))

    # ---- Fit controls: quantify what the two scalars depend on ----------------
    report = None
    if (args.fit_report or args.st_modes) and all_rel:
        n_anchor = sum(len(x) for x in all_rel)
        st_all = _fit_scale_shift(np.concatenate(all_rel), np.concatenate(all_zr), np.concatenate(all_cf))
        st_train = _fit_subset(all_rel, all_zr, all_cf, idx=tr_idx) if tr_idx is not None else None
        med_rel = float(np.median(rel_meds_train)) if rel_meds_train else None
        diag = None
        if cam_pos_train:
            P = np.stack(cam_pos_train)
            diag = float(np.linalg.norm(P.max(0) - P.min(0)))
        report = {
            "tag": args.tag, "n_views": len(names),
            "n_train_views": int(sum(1 for nm in names if n2split.get(nm, "train") != "test")),
            "n_anchor_px": int(n_anchor),
            "n_anchor_px_train": int(len(tr_idx)) if tr_idx is not None else 0,
            "st_all": st_all, "st_train": st_train,
            "median_rel_train": med_rel, "traj_bbox_diag_m": diag,
            "st_fix3m": ((3.0 / med_rel, 0.0) if med_rel else None),
            "st_bbox": ((diag / 2.0 / med_rel, 0.0) if (med_rel and diag) else None),
            "k_fits": {},
        }
        for k in (1, 5, 10, 30, 100):
            report["k_fits"][str(k)] = [
                _fit_subset(all_rel, all_zr, all_cf, idx=tr_idx, k=k, seed=s)
                for s in range(5)]
        if args.fit_report:
            os.makedirs(os.path.dirname(args.fit_report) or ".", exist_ok=True)
            json.dump(report, open(args.fit_report, "w"), indent=1)
            print(f"[fit-report] {args.tag}: st_all={st_all} st_train={st_train} "
                  f"anchors={n_anchor} ({report['n_anchor_px_train']} train) -> {args.fit_report}")

    # ---- Rule-based / subsampled (s,b) prior arms (one DA-V2 pass, many arms) --
    if args.st_modes:
        mode_st = {}
        for md in args.st_modes.split(","):
            md = md.strip()
            if not md:
                continue
            if md == "trainfit":
                mode_st[md] = report["st_train"]
            elif md == "fix3m":
                mode_st[md] = report["st_fix3m"]
            elif md == "bbox":
                mode_st[md] = report["st_bbox"]
            elif md.startswith("k"):
                mode_st[md] = report["k_fits"][md[1:]][0]   # seed-0 draw
            else:
                sys.exit(f"unknown st-mode {md}")
        for md, st in mode_st.items():
            if st is None:
                print(f"[st-mode {md}] {args.tag}: no fit, skipped"); continue
            od = f"{args.st_out_base or os.path.join('outputs/anchored_depth', args.tag)}_{md}"
            os.makedirs(od, exist_ok=True)
            nsv = 0
            for nm, rel, rng, inten, lid in per:
                if rel is None:
                    continue
                depth = np.clip(st[0] * rel + st[1], 0.05, 30.0).astype(np.float32)
                np.savez_compressed(os.path.join(od, nm + ".npz"),
                                    depth=depth, conf=np.ones_like(depth, np.float32))
                nsv += 1
            print(f"[st-mode {md}] {args.tag}: (s,b)=({st[0]:.4f},{st[1]:.4f}) saved {nsv} -> {od}")
        return
    # 'lidarscale': DA-V2 rel-depth + GLOBAL scale-shift fit on LiDAR pixels
    # (the LiDAR analog of 'global' — substitution upper bound for init clouds).
    if args.mode == "lidarscale":
        rl, zl = [], []
        for nm, rel, rng, inten, lid in per:
            m = lid > 0
            if rel is not None and m.any():
                rl.append(rel[m]); zl.append(lid[m])
        if not rl:
            sys.exit("lidarscale: no lidar pixels")
        g_st = _fit_scale_shift(np.concatenate(rl), np.concatenate(zl),
                                np.ones(sum(len(x) for x in rl), np.float32))

    os.makedirs(args.out, exist_ok=True)
    n_saved = 0
    n_fallback = 0
    g_err, r_err, base_n = [], [], 0
    for nm, rel, rng, inten, lid in per:
        if args.mode == "lidar":
            depth = lid; conf = (lid > 0).astype(np.float32)
        else:
            n_anc = int((inten > 0).sum())
            st_r = (_fit_scale_shift(rel, rng, inten)
                    if n_anc >= args.perview_min_anchors else None)  # per-view radar fit
            st = (st_r if args.mode == "radar" else g_st)
            if args.mode == "radar" and st_r is None:
                n_fallback += 1
            if st is None:
                st = g_st if g_st is not None else (1.0, 0.0)
            depth = (st[0] * rel + st[1]).astype(np.float32)
            depth = np.clip(depth, 0.05, 30.0)
            if getattr(args, "conf_from_anchors", False):
                # uncertainty-weighted supervision (review response B3.2):
                # per-pixel confidence = blurred CFAR anchor intensity/density,
                # normalized to (0.05, 1]
                from scipy.ndimage import gaussian_filter
                cmap = gaussian_filter(inten.astype(np.float32), 24.0)
                mx = float(cmap.max())
                conf = ((0.05 + 0.95 * cmap / mx).astype(np.float32)
                        if mx > 0 else np.full_like(depth, 0.05, np.float32))
            else:
                conf = np.ones_like(depth, np.float32)
            # gate: error vs lidar GT (where lidar valid)
            lm = lid > 0
            if lm.any() and st_r is not None and g_st is not None:
                base_n += int(lm.sum())
                dg = g_st[0] * rel + g_st[1]; dr = st_r[0] * rel + st_r[1]
                g_err.append(np.abs(dg[lm] - lid[lm]).sum())
                r_err.append(np.abs(dr[lm] - lid[lm]).sum())
        if not args.gate:
            np.savez_compressed(os.path.join(args.out, nm + ".npz"), depth=depth, conf=conf)
            n_saved += 1

    if g_err and base_n:
        print(f"[GATE] {args.tag}: depth MAE vs LiDAR  global-align={sum(g_err)/base_n:.3f}m  "
              f"radar-align={sum(r_err)/base_n:.3f}m  (lower=better; radar should win if it adds metric info)")
    if not args.gate:
        if args.mode == "radar":
            print(f"[radar] {args.tag}: per-view fits, {n_fallback}/{n_saved or len(per)} "
                  f"views fell back to the global fit (<{args.perview_min_anchors} anchors)")
        print(f"[{args.mode}] {args.tag}: saved {n_saved} priors -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="radar", choices=["global", "radar", "lidar", "sfm", "lidarscale"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="only print depth-vs-lidar gate, do not save")
    ap.add_argument("--snapshot", action="store_true", help="use old single-t_tar LIRA snapshot anchors (debug)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="darkness level: corrupt RGB before DA-V2 (match training; <1.0 only)")
    ap.add_argument("--fit-split", default="all", choices=["all", "train"],
                    help="which views' anchors enter the global scale-shift fit")
    ap.add_argument("--fit-report", default=None,
                    help="dump JSON with fit controls (all vs train-only vs K-sample, rule scales)")
    ap.add_argument("--st-modes", default=None,
                    help="comma list of rule-based (s,b) prior arms to save: trainfit,fix3m,bbox,k1,k5,k10")
    ap.add_argument("--st-out-base", default=None,
                    help="output dir base for --st-modes (default outputs/anchored_depth/<tag>)")
    ap.add_argument("--exclude-test-radar", default=None,
                    help="Strict-protocol control: exclude radar frames near test "
                         "timestamps ('pf' = nearest frame per test time, or a "
                         "window in seconds, e.g. 0.5 / 2.0).")
    ap.add_argument("--conf-from-anchors", action="store_true",
                    help="write per-pixel confidence = blurred CFAR anchor "
                         "intensity (uncertainty-weighted supervision control)")
    ap.add_argument("--perview-min-anchors", type=int, default=8,
                    help="--mode radar: min anchor pixels for a per-view fit; "
                         "below this the view falls back to the global fit.")
    ap.add_argument("--da-model", default=None,
                    help="override monocular backbone (HF id), e.g. "
                         "depth-anything/Depth-Anything-V2-Base-hf")
    args = ap.parse_args()
    if args.da_model:
        global _DA_MODEL
        _DA_MODEL = args.da_model
    if args.out is None:
        args.out = f"outputs/anchored_depth/{args.tag}_{args.mode}"
    run(args)


if __name__ == "__main__":
    main()
