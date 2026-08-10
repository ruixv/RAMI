"""Abstract Baseline interface + registry for Phase 2 head-to-head runs.

Every baseline (B1-B20) implements this interface. The training entrypoint
(`training/baseline_train.py`) instantiates the registered baseline class,
calls `.train()` on the COLMAP-exported scene, and then `.render()` at the
held-out viewpoints for `eval/render_and_score.py` to grade.

A baseline can be either:
  * a *native* implementation in pure PyTorch (overrides train/render),
  * or a *subprocess wrapper* around an upstream repo (overrides
    `train_subprocess` / `render_subprocess`).

For now we declare the metadata (`BaselineMeta`) and the interface; concrete
implementations are stubbed in their own files and raise NotImplementedError
with explicit setup instructions until the gsplat env compiles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Type

import numpy as np


@dataclass(frozen=True)
class BaselineMeta:
    """Static metadata about a baseline."""

    key: str           # short id, e.g. "vanilla_3dgs"
    paper_id: str      # citation key for the paper
    venue: str         # e.g. "SIGGRAPH 2023"
    tier: int          # 1-4, see docs/eval_protocol.md
    requires_gsplat: bool = False
    requires_native_cuda: bool = False
    upstream_repo: str = ""        # URL or path to upstream code
    notes: str = ""


@dataclass
class TrainResult:
    """Lightweight handle returned from Baseline.train()."""

    checkpoint_dir: str
    n_iters: int
    seconds_elapsed: float
    extra: dict = field(default_factory=dict)


class Baseline(ABC):
    """Subclass and call `register(cls)` (or use `@register` decorator)."""

    meta: BaselineMeta

    @abstractmethod
    def train(
        self,
        export_dir: str,
        output_dir: str,
        config: dict,
    ) -> TrainResult:
        """Fit the baseline to the COLMAP-exported scene at `export_dir`,
        save checkpoint(s) under `output_dir`, and return a TrainResult."""

    @abstractmethod
    def render(
        self,
        checkpoint_dir: str,
        w2c_matrices: np.ndarray,
        K: np.ndarray,
        image_w: int,
        image_h: int,
    ) -> np.ndarray:
        """Render each input view. Returns float32 array
        (n_views, H, W, 3) in [0, 1] sRGB."""


_REGISTRY: dict[str, Type[Baseline]] = {}


def register(cls: Type[Baseline]) -> Type[Baseline]:
    if not hasattr(cls, "meta") or not isinstance(cls.meta, BaselineMeta):
        raise TypeError(f"{cls.__name__} must define class attribute `meta: BaselineMeta`")
    if cls.meta.key in _REGISTRY:
        raise ValueError(f"baseline key {cls.meta.key!r} already registered by "
                         f"{_REGISTRY[cls.meta.key].__name__}")
    _REGISTRY[cls.meta.key] = cls
    return cls


def get_baseline(key: str) -> Type[Baseline]:
    if key not in _REGISTRY:
        raise KeyError(f"unknown baseline key {key!r}; registered: {list(_REGISTRY.keys())}")
    return _REGISTRY[key]


def list_baselines() -> list[BaselineMeta]:
    return [cls.meta for cls in _REGISTRY.values()]


# --------- shared helper for stub baselines ---------


class _NotImplementedBaseline(Baseline):
    """Mixin used by stubs while we wait for the gsplat env to build.

    Concrete stubs set `meta = BaselineMeta(...)` and inherit; both train /
    render raise NotImplementedError with the doc pointer."""

    def train(self, export_dir, output_dir, config):
        raise NotImplementedError(
            f"baseline {self.meta.key!r} not yet runnable on this server.\n"
            f"See docs/phase2_env.md for the gsplat build status. Once a "
            f"working GCC ≥ 9 is available, port the upstream train script "
            f"from {self.meta.upstream_repo or '(repo TBD)'}."
        )

    def render(self, checkpoint_dir, w2c_matrices, K, image_w, image_h):
        raise NotImplementedError(
            f"baseline {self.meta.key!r} render() — same install dependency as train()"
        )
