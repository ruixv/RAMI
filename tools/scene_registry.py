"""Benchmark scene registry: tag -> RadarEyes capture directory name.

The 23-scene benchmark of the paper. Capture directories are expected under
$RAMI_DATA_ROOT (default ./data). See DATA.md for the per-scene catalog.
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, os.pardir, "configs", "scenes.json")) as _f:
    SCENES = json.load(_f)

__all__ = ["SCENES"]
