#!/usr/bin/env python3
"""Aggregate metrics of finished runs under outputs/runs/ into one table.

Each run directory <tag>_<arm>/ must contain metrics_vs_clean.json (written
by tools/score_vs_clean.py). Prints per-scene PSNR/SSIM/LPIPS per arm plus
the mean delta of every arm against the random baseline where available.

Usage: python tools/aggregate_results.py [--runs outputs/runs]
"""
import argparse
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/runs")
    args = ap.parse_args()
    by = defaultdict(dict)  # tag -> arm -> mean metrics
    if not os.path.isdir(args.runs):
        print(f"no runs directory at {args.runs}")
        return
    for d in sorted(os.listdir(args.runs)):
        p = os.path.join(args.runs, d, "metrics_vs_clean.json")
        if not os.path.exists(p) or "_" not in d:
            continue
        tag, _, arm = d.rpartition("_")
        by[tag][arm] = json.load(open(p))["mean"]
    if not by:
        print("no finished runs found")
        return
    arms = sorted({a for v in by.values() for a in v})
    hdr = "scene".ljust(12) + "".join(f"{a:>24}" for a in arms)
    print(hdr)
    print(" " * 12 + "".join(f"{'PSNR   SSIM   LPIPS':>24}" for _ in arms))
    for tag in sorted(by):
        cells = []
        for a in arms:
            m = by[tag].get(a)
            cells.append(f"{m['psnr']:6.2f} {m['ssim']:6.4f} {m['lpips']:6.4f}"
                         if m else " " * 20)
        print(tag.ljust(12) + "".join(f"{c:>24}" for c in cells))
    if "random" in arms:
        for a in arms:
            if a == "random":
                continue
            ds = [by[t][a]["psnr"] - by[t]["random"]["psnr"]
                  for t in by if a in by[t] and "random" in by[t]]
            if ds:
                print(f"mean PSNR delta {a} - random: {sum(ds)/len(ds):+.2f} dB "
                      f"({len(ds)} scenes)")


if __name__ == "__main__":
    main()
