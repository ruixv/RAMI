"""RadarDepthNet — the fail-fast GATE for the learned-fusion hypothesis (2026-06-07).

Question: does sparse+noisy radar carry depth information a LEARNED net can use,
beyond what RGB alone gives? We train a small U-Net (RGB [+radar] -> dense depth
+ aleatoric uncertainty), self-supervised on projected LiDAR depth, and compare
WITH vs WITHOUT the radar input channels on HELD-OUT scenes (leave-scenes-out).
If with-radar reduces held-out depth error -> radar adds usable info -> proceed
to 3DGS integration. If not -> the information ceiling holds at the fusion level.

Modes:
  cache  --scenes a,b,c   : precompute per-view {RGB, LiDAR-depth, radar-maps}
  train  --train s1,..  --test t1,..  [--no-radar]  : train + eval, print depth err

See docs/learned_radar_fusion_design.md.
"""
from __future__ import annotations
import argparse, os, sys, json, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from data.radareyes import Scene
from data.radareyes.sync import interpolate_pose, nearest_index
from data.radareyes.transforms import SENSOR_OFFSETS, sensor_to_world
from tools.zed_calibration import ZED_BODY_TO_OPTICAL_R
from models.baselines._gsplat_common import K_from_camera, parse_colmap_export
from models.baselines.dn_splatter import _build_lidar_depth_for_view, _resolve_scene_from_splits
from models.baselines.radarsplat_physics import _build_lira_with_intensity
from tools.fov_mask import project_to_camera
from tools.scene_registry import SCENES

CACHE = "outputs/completion_cache"
HW = (180, 320)  # low-res H,W


def _radar_maps(anchors, weights, t_img, scene, Kl, W, H, dev):
    rng = np.zeros((H, W), np.float32); inten = np.zeros((H, W), np.float32)
    zp, zq = interpolate_pose(scene.zed.pose_stream, t_img)
    fx, fy, cx, cy = float(Kl[0, 0]), float(Kl[1, 1]), float(Kl[0, 2]), float(Kl[1, 2])
    px, dp, va = project_to_camera(anchors.astype(np.float64), zp, zq, fx=fx, fy=fy, cx=cx, cy=cy,
                                   image_w=W, image_h=H, near=0.3, far=20.0,
                                   body_to_optical_R=ZED_BODY_TO_OPTICAL_R)
    if va.any():
        u = px[va, 0].astype(int).clip(0, W - 1); v = px[va, 1].astype(int).clip(0, H - 1)
        d = dp[va].astype(np.float32); w = weights[va].astype(np.float32)
        for ui, vi, di, wi in zip(u, v, d, w):
            # splat 3x3 (radar's coarse angular footprint), keep nearest
            for dv in (-1, 0, 1):
                for du in (-1, 0, 1):
                    yy, xx = vi + dv, ui + du
                    if 0 <= yy < H and 0 <= xx < W and (rng[yy, xx] == 0 or di < rng[yy, xx]):
                        rng[yy, xx] = di; inten[yy, xx] = wi
    return rng, inten


def _ramap_from_adc(adc):
    """Dense range-azimuth map (128 range x 8 azimuth) from one raw ADC frame —
    RadarBoostVLM's proven geometry-carrying representation (radar->lidar AP 0.31
    on our data vs 0.27 sparse). |cube| max over doppler & elevation, log1p+std."""
    rfft = np.fft.fft(adc, 128, axis=0)
    dfft = np.fft.fftshift(np.fft.fft(rfft, 128, axis=1), axes=1)
    mimo = np.zeros((128, 128, 8, 2), dtype=dfft.dtype)
    mimo[:, :, 0:4, 0] = dfft[:, :, :, 0]; mimo[:, :, 4:8, 0] = dfft[:, :, :, 2]; mimo[:, :, 2:6, 1] = dfft[:, :, :, 1]
    azi = np.fft.fftshift(np.fft.fft(mimo, 8, axis=2), axes=2)
    cube = np.abs(np.fft.fftshift(np.fft.fft(azi, 8, axis=3), axes=3))
    ra = cube.max(axis=(1, 3))
    ra = np.log1p(ra); ra = (ra - ra.mean()) / (ra.std() + 1e-8)
    return ra.astype(np.float32)  # (128, 8)


def _view_ramap(scene, t):
    """RA-map of the radar_azi frame nearest in time to camera-view time t."""
    idx = int(nearest_index(scene.radar_azi.timestamps, t))
    idx = max(0, min(idx, len(scene.radar_azi) - 1))
    return _ramap_from_adc(scene.radar_azi.load_adc(idx))


def cmd_cache(args):
    dev = "cuda"; os.makedirs(CACHE, exist_ok=True)
    H, W = HW
    for tag in args.scenes.split(","):
        tag = tag.strip()
        out = os.path.join(CACHE, f"{tag}.npz")
        if os.path.exists(out) and not args.force:
            print(f"{tag}: cached, skip"); continue
        rd = f"outputs/canonical_gt/{tag}"
        ed = rd  # canonical_gt dir holds splits.json (scene, intrinsics, records)
        if not os.path.exists(os.path.join(ed, "splits.json")):
            print(f"{tag}: no splits.json in {ed}; skip"); continue
        splits = json.load(open(os.path.join(ed, "splits.json")))
        intr = splits["intrinsics"]; Wf, Hf = splits["image_size"]
        Kf = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], np.float64)
        sx, sy = W / Wf, H / Hf
        Kl = Kf.copy(); Kl[0, :] *= sx; Kl[1, :] *= sy
        Klt = torch.tensor(Kl, dtype=torch.float32, device=dev)
        scene = _resolve_scene_from_splits(ed)
        n2t = {r["filename"]: r["t_image"] for r in splits["records"]}
        test_names = set(splits.get("test_image_names", []))
        t_tar = float(np.median(scene.radar_azi.timestamps))
        anc, w = _build_lira_with_intensity(scene, t_tar, {})
        recs = [r for r in splits["records"] if r["filename"] in n2t]
        RGB, DEP, RAD, IST, RAM, NAMES = [], [], [], [], [], []
        for r in recs:
            nm = r["filename"]; t = float(n2t[nm])
            img = np.asarray(Image.open(os.path.join(rd, "images", nm)).convert("RGB").resize((W, H)), np.uint8)
            dlow = _build_lidar_depth_for_view(scene, t, Klt, W, H, device=dev).cpu().numpy().astype(np.float16)
            rmap, imap = _radar_maps(anc, w, t, scene, Kl, W, H, dev)
            RGB.append(img); DEP.append(dlow); RAD.append(rmap.astype(np.float16)); IST.append(imap.astype(np.float16))
            RAM.append(_view_ramap(scene, t).astype(np.float16))  # dense RA-map (128,8)
            NAMES.append(nm)
        np.savez_compressed(out, rgb=np.stack(RGB), depth=np.stack(DEP), radar_rng=np.stack(RAD),
                            radar_int=np.stack(IST), ramap=np.stack(RAM), names=np.array(NAMES),
                            is_test=np.array([n in test_names for n in NAMES]))
        cov = float((np.stack(DEP) > 0).mean()); rcov = float((np.stack(RAD) > 0).mean())
        print(f"{tag}: cached {len(NAMES)} views  lidar_cov={cov*100:.2f}%  radar_cov={rcov*100:.2f}% -> {out}")


class UNet(nn.Module):
    def __init__(self, in_ch, use_ramap=False):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.GroupNorm(8, o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.ReLU())
        self.e1 = blk(in_ch, 32); self.e2 = blk(32, 64); self.e3 = blk(64, 128)
        self.b = blk(128, 128)
        self.d3 = blk(128 + 128, 64); self.d2 = blk(64 + 64, 32); self.d1 = blk(32 + 32, 32)
        self.head = nn.Conv2d(32, 2, 1)  # depth, log-var
        self.pool = nn.MaxPool2d(2)
        # Dense RA-map radar branch (RadarBoostVLM repr): polar (1,128,8) -> 128-d
        # global feature, injected (FiLM-style add) into the U-Net bottleneck.
        self.use_ramap = use_ramap
        if use_ramap:
            self.radar = nn.Sequential(
                nn.Conv2d(1, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.GELU(),
                nn.AdaptiveAvgPool2d(1))
            self.radar_proj = nn.Linear(128, 128)

    @staticmethod
    def _up(x, skip):
        return F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x, ram=None):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        if self.use_ramap and ram is not None:
            rv = self.radar(ram).flatten(1)            # (B,128)
            rv = self.radar_proj(rv)[:, :, None, None]  # (B,128,1,1)
            b = b + rv                                  # inject radar geometry at bottleneck
        d3 = self.d3(torch.cat([self._up(b, e3), e3], 1))
        d2 = self.d2(torch.cat([self._up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([self._up(d2, e1), e1], 1))
        o = self.head(d1)
        return F.softplus(o[:, :1]), o[:, 1:]  # depth>0, logvar


def _load(tags):
    # Materialize each compressed array into RAM ONCE (npz[key] re-decompresses
    # the WHOLE array on every access -> 100x slowdown if done per-iter).
    X = []
    for t in tags:
        p = os.path.join(CACHE, f"{t}.npz")
        if not os.path.exists(p): print(f"warn: no cache {t}"); continue
        z = np.load(p, allow_pickle=True)
        d = {"rgb": z["rgb"], "depth": z["depth"], "radar_rng": z["radar_rng"],
             "radar_int": z["radar_int"], "names": z["names"]}
        if "ramap" in z.files:
            d["ramap"] = z["ramap"]
        z.close()
        X.append((t, d))
    return X


def _batch(npz, idxs, mode, dev):
    """mode: 'none' (RGB), 'sparse' (RGB+sparse radar maps), 'ramap' (RGB + dense
    RA-map branch). Returns (inp, ram, dep); ram is None unless mode=='ramap'."""
    rgb = torch.tensor(npz["rgb"][idxs], dtype=torch.float32, device=dev).permute(0, 3, 1, 2) / 255.0
    dep = torch.tensor(npz["depth"][idxs].astype(np.float32), device=dev).unsqueeze(1)
    ram = None
    if mode == "sparse":
        rr = torch.tensor(npz["radar_rng"][idxs].astype(np.float32), device=dev).unsqueeze(1)
        ri = torch.tensor(npz["radar_int"][idxs].astype(np.float32), device=dev).unsqueeze(1)
        inp = torch.cat([rgb, rr / 10.0, ri], 1)
    else:
        inp = rgb
        if mode == "ramap":
            ram = torch.tensor(npz["ramap"][idxs].astype(np.float32), device=dev).unsqueeze(1)  # (B,1,128,8)
    return inp, ram, dep


def cmd_train(args):
    dev = "cuda"; torch.manual_seed(args.seed); np.random.seed(args.seed)
    mode = "none" if args.no_radar else args.radar_mode  # none | sparse | ramap
    tr = _load(args.train.split(",")); te = _load(args.test.split(","))
    in_ch = 5 if mode == "sparse" else 3
    net = UNet(in_ch, use_ramap=(mode == "ramap")).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    pool = [(si, vi) for si, (t, d) in enumerate(tr) for vi in range(len(d["names"]))]
    print(f"[train] mode={mode} in_ch={in_ch}  train_views={len(pool)} ({len(tr)} scenes)  test={len(te)} scenes")
    B = 8
    for it in range(1, args.iters + 1):
        sel = [pool[k] for k in np.random.randint(0, len(pool), B)]
        inp_l, ram_l, dep_l = [], [], []
        for si, vi in sel:
            inp, ram, dep = _batch(tr[si][1], [vi], mode, dev)
            inp_l.append(inp); dep_l.append(dep)
            if ram is not None: ram_l.append(ram)
        inp = torch.cat(inp_l); dep = torch.cat(dep_l)
        ram = torch.cat(ram_l) if ram_l else None
        pd, lv = net(inp, ram)
        m = (dep > 0).float()
        err = (torch.log(pd.clamp(0.1, 25)) - torch.log(dep.clamp(0.1, 25)))
        nll = (0.5 * torch.exp(-lv) * err**2 + 0.5 * lv) * m
        loss = nll.sum() / (m.sum() + 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == 1:
            print(f"  it{it} loss={loss.item():.4f}")
    net.eval(); maes, rmses, n = [], [], 0
    with torch.no_grad():
        for t, d in te:
            for vi in range(len(d["names"])):
                inp, ram, dep = _batch(d, [vi], mode, dev)
                pd, _ = net(inp, ram); m = dep > 0
                if m.sum() < 20: continue
                e = (pd[m] - dep[m]).abs()
                maes.append(e.mean().item()); rmses.append((e**2).mean().item()**0.5); n += 1
    MAE = float(np.mean(maes)) if maes else float("nan")
    RMSE = float(np.mean(rmses)) if rmses else float("nan")
    print(f"\n=== [mode={mode}] held-out depth error over {n} views: MAE={MAE:.3f}m  RMSE={RMSE:.3f}m ===")
    res = {"mode": mode, "MAE": MAE, "RMSE": RMSE, "n_views": n,
           "train": args.train, "test": args.test, "iters": args.iters, "seed": args.seed}
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, f"result_{mode}_s{args.seed}.json"), "w") as f:
        json.dump(res, f, indent=2)
    if args.save:
        torch.save({"state": net.state_dict(), "in_ch": in_ch, "mode": mode}, args.save)
        print(f"saved net -> {args.save}")


PRIORS = "outputs/depth_priors"


def cmd_infer(args):
    """Produce full-res per-view dense depth + confidence priors for a scene's
    TRAIN views, for use as 3DGS depth supervision. conf = 1/exp(logvar)."""
    dev = "cuda"
    ck = torch.load(args.weights, map_location=dev)
    mode = ck.get("mode", "sparse" if ck.get("use_radar") else "none")
    net = UNet(ck["in_ch"], use_ramap=(mode == "ramap")).to(dev); net.load_state_dict(ck["state"]); net.eval()
    d = _load([args.scene])[0][1]
    splits = json.load(open(f"outputs/canonical_gt/{args.scene}/splits.json"))
    Wf, Hf = splits["image_size"]
    train_names = set(r["filename"] for r in splits["records"] if r.get("split") != "test")
    variant = {"none": "rgb", "sparse": "radar", "ramap": "radar_ramap"}[mode]
    outdir = os.path.join(PRIORS, args.scene, variant)
    os.makedirs(outdir, exist_ok=True)
    n = 0
    with torch.no_grad():
        for vi, nm in enumerate(d["names"]):
            if str(nm) not in train_names:
                continue
            inp, ram, _ = _batch(d, [vi], mode, dev)
            pd, lv = net(inp, ram)
            depth = F.interpolate(pd, size=(Hf, Wf), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float16)
            conf = F.interpolate(torch.exp(-lv), size=(Hf, Wf), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float16)
            np.savez_compressed(os.path.join(outdir, f"{str(nm)}.npz"), depth=depth, conf=conf)
            n += 1
    print(f"infer {args.scene} [{variant}]: {n} train-view priors -> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache"); c.add_argument("--scenes", required=True); c.add_argument("--force", action="store_true")
    t = sub.add_parser("train"); t.add_argument("--train", required=True); t.add_argument("--test", required=True)
    t.add_argument("--no-radar", action="store_true"); t.add_argument("--iters", type=int, default=3000)
    t.add_argument("--radar-mode", default="sparse", choices=["none", "sparse", "ramap"])
    t.add_argument("--seed", type=int, default=0); t.add_argument("--save", default=None)
    i = sub.add_parser("infer"); i.add_argument("--weights", required=True); i.add_argument("--scene", required=True)
    a = ap.parse_args()
    {"cache": cmd_cache, "train": cmd_train, "infer": cmd_infer}[a.cmd](a)
