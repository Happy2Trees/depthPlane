from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    input_mesh: Path
    output_mesh: Path
    samples: int
    fit_threshold: float
    remove_threshold: float | None
    bottom_fraction: float
    plane_extreme: str
    passes: int
    clip_halfspace: bool
    slab_thickness: float | None
    min_normal_dot: float


DEFAULTS: dict[str, Any] = {
    "input_mesh": "data/only_structure.stl",
    "output_mesh": "output/out_no_floor.stl",
    "samples": 150_000,
    "fit_threshold": 0.03,
    "remove_threshold": None,
    "bottom_fraction": 0.05,
    "plane_extreme": "bottom",
    "passes": 2,
    "clip_halfspace": False,
    "slab_thickness": None,
    "min_normal_dot": 0.98,
}

ALLOWED_EXTREMES = {"bottom", "top", "auto"}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def load_config(path: Path) -> Config:
    """Load YAML config, apply defaults, and validate fields."""

    raw = {**DEFAULTS, **_read_yaml(path)}

    if raw["plane_extreme"] not in ALLOWED_EXTREMES:
        raise ValueError(
            f"plane_extreme must be one of {sorted(ALLOWED_EXTREMES)}, "
            f"got '{raw['plane_extreme']}'",
        )

    return Config(
        input_mesh=Path(str(raw["input_mesh"])),
        output_mesh=Path(str(raw["output_mesh"])),
        samples=int(raw["samples"]),
        fit_threshold=float(raw["fit_threshold"]),
        remove_threshold=(
            None if raw["remove_threshold"] is None else float(raw["remove_threshold"])
        ),
        bottom_fraction=float(raw["bottom_fraction"]),
        plane_extreme=str(raw["plane_extreme"]),
        passes=int(raw["passes"]),
        clip_halfspace=bool(raw["clip_halfspace"]),
        slab_thickness=(
            None if raw["slab_thickness"] is None else float(raw["slab_thickness"])
        ),
        min_normal_dot=float(raw["min_normal_dot"]),
    )
