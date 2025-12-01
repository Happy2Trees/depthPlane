from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import laspy
import matplotlib.pyplot as plt
import numpy as np

Statistic = Literal["min", "max", "mean"]
OutlierMethod = Literal["clip", "iqr", "none"]


@dataclass
class DepthMapResult:
    """Container for a computed depth map."""

    depth_mm: np.ndarray
    resolution: float
    plane_z: float
    bbox: tuple[float, float, float, float]
    clip_min: float | None = None
    clip_max: float | None = None
    outlier_method: OutlierMethod = "none"
    iqr_multiplier: float | None = None
    iqr_sample_count: int | None = None

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) for plotting."""

        x_min, y_min, x_max, y_max = self.bbox
        return (x_min, x_max, y_min, y_max)


def _reservoir_sample_z_values(
    las_path: Path,
    chunk_size: int,
    sample_size: int | None,
    rng: np.random.Generator,
    progress: bool,
) -> np.ndarray:
    """Return a reservoir sample of z values to estimate IQR bounds."""

    if sample_size is None or sample_size <= 0:
        collected: list[np.ndarray] = []
        with laspy.open(las_path) as reader:
            for chunk in reader.chunk_iterator(chunk_size):
                vals = np.asarray(chunk.z, dtype=np.float64)
                if vals.size:
                    collected.append(vals)
        if not collected:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(collected)

    sample = np.empty(sample_size, dtype=np.float64)
    filled = 0
    total_seen = 0

    with laspy.open(las_path) as reader:
        for chunk_idx, chunk in enumerate(reader.chunk_iterator(chunk_size), start=1):
            vals = np.asarray(chunk.z, dtype=np.float64)
            if vals.size == 0:
                continue

            n = vals.size
            if filled < sample_size:
                take = min(sample_size - filled, n)
                sample[filled:filled + take] = vals[:take]
                filled += take
                total_seen += take
                vals = vals[take:]
                n = vals.size

            if n:
                indices = np.arange(total_seen + 1, total_seen + n + 1, dtype=np.int64)
                probs = sample_size / indices.astype(np.float64)
                keep_mask = rng.random(n) < probs
                if keep_mask.any():
                    replace_idx = rng.integers(0, sample_size, size=int(keep_mask.sum()))
                    sample[replace_idx] = vals[keep_mask]
                total_seen += n

            if progress and chunk_idx % 50 == 0:
                print(
                    f"[depth-map] IQR sampling: filled {min(filled, sample_size):,}/"
                    f"{sample_size:,} (seen {total_seen:,} points, chunk {chunk_idx})",
                )

    if filled == 0:
        return np.empty(0, dtype=np.float64)
    if filled < sample_size:
        return sample[:filled]
    return sample


def _compute_iqr_bounds(
    las_path: Path,
    chunk_size: int,
    whisker_scale: float,
    sample_size: int | None,
    progress: bool,
) -> tuple[float, float, int]:
    """Estimate lower/upper bounds using the IQR rule from a z-value sample."""

    if whisker_scale < 0:
        raise ValueError("iqr_multiplier must be non-negative.")

    rng = np.random.default_rng(42)
    samples = _reservoir_sample_z_values(
        las_path=las_path,
        chunk_size=chunk_size,
        sample_size=sample_size,
        rng=rng,
        progress=progress,
    )
    if samples.size == 0:
        raise ValueError("No z-values found; unable to compute IQR bounds.")

    q1, q3 = np.quantile(samples, [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - whisker_scale * iqr
    upper = q3 + whisker_scale * iqr
    return float(lower), float(upper), int(samples.size)


def _init_accumulator(
    statistic: Statistic,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Allocate accumulator arrays based on the statistic."""

    if statistic == "min":
        return np.full(shape, np.inf, dtype=np.float32), None
    if statistic == "max":
        return np.full(shape, -np.inf, dtype=np.float32), None
    if statistic == "mean":
        return (
            np.zeros(shape, dtype=np.float64),
            np.zeros(shape, dtype=np.uint32),
        )
    raise ValueError(f"Unsupported statistic '{statistic}'")


def _update_accumulator(
    statistic: Statistic,
    accumulator: np.ndarray,
    helper: np.ndarray | None,
    xi: np.ndarray,
    yi: np.ndarray,
    values: np.ndarray,
) -> None:
    """Scatter-update the accumulator with chunked point data."""

    if statistic == "min":
        np.minimum.at(accumulator, (yi, xi), values)
    elif statistic == "max":
        np.maximum.at(accumulator, (yi, xi), values)
    else:  # mean
        assert helper is not None
        np.add.at(accumulator, (yi, xi), values)
        np.add.at(helper, (yi, xi), 1)


def _finalize_depth(
    statistic: Statistic,
    accumulator: np.ndarray,
    helper: np.ndarray | None,
) -> np.ndarray:
    """Convert accumulator to a finalized depth array with NaNs for empty bins."""

    if statistic == "mean":
        assert helper is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = accumulator / helper
            depth[helper == 0] = np.nan
        return depth.astype(np.float32)

    depth = accumulator
    if statistic == "min":
        depth[depth == np.inf] = np.nan
    elif statistic == "max":
        depth[depth == -np.inf] = np.nan
    return depth


def generate_depth_map(
    las_path: Path,
    resolution: float = 0.01,
    plane_z: float = 0.0,
    statistic: Statistic = "min",
    chunk_size: int = 2_000_000,
    progress: bool = True,
    outlier_method: OutlierMethod = "clip",
    clip_min: float | None = None,
    clip_max: float | None = None,
    iqr_multiplier: float = 1.5,
    iqr_sample_size: int | None = 500_000,
) -> DepthMapResult:
    """Generate an orthogonal projection depth map from a LAS file.

    Parameters
    ----------
    las_path:
        Path to the input LAS file.
    resolution:
        Grid resolution in meters per pixel (default: 1 cm).
    plane_z:
        Reference plane height (meters). Depth values are measured from this plane.
    statistic:
        Aggregation function for multiple points falling into one pixel:
        "min", "max", or "mean" (default: "min").
    chunk_size:
        Number of points to process per chunk to keep memory bounded.
    progress:
        Whether to print progress information.
    outlier_method:
        Which outlier filter to apply: "clip" uses clip_min/clip_max bounds,
        "iqr" estimates bounds via the IQR rule, and "none" disables filtering.
    clip_min / clip_max:
        Optional z-value bounds (meters). Used when outlier_method is "clip".
    iqr_multiplier:
        Whisker coefficient k for the IQR rule (bounds: Q1 - k*IQR, Q3 + k*IQR).
    iqr_sample_size:
        Reservoir sample size used to estimate IQR bounds. Set to None or <=0 to
        use all points (may increase memory/time).
    """

    if resolution <= 0:
        raise ValueError("resolution must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if outlier_method not in {"clip", "iqr", "none"}:
        raise ValueError("outlier_method must be one of: clip, iqr, none.")
    las_path = las_path.expanduser()
    if not las_path.exists():
        raise FileNotFoundError(las_path)

    filter_min = clip_min if outlier_method == "clip" else None
    filter_max = clip_max if outlier_method == "clip" else None
    iqr_sample_count: int | None = None

    if outlier_method == "iqr":
        filter_min, filter_max, iqr_sample_count = _compute_iqr_bounds(
            las_path=las_path,
            chunk_size=chunk_size,
            whisker_scale=iqr_multiplier,
            sample_size=iqr_sample_size,
            progress=progress,
        )
        if progress:
            msg = (
                "[depth-map] IQR bounds ("
                f"k={iqr_multiplier}, samples={iqr_sample_count:,}): "
                f"{filter_min:.3f} m to {filter_max:.3f} m"
            )
            print(msg)

    if filter_min is not None and filter_max is not None and filter_min > filter_max:
        raise ValueError("clip_min cannot be greater than clip_max.")

    with laspy.open(las_path) as reader:
        mins = reader.header.mins
        maxs = reader.header.maxs
        x_min, y_min, _ = mins
        x_max, y_max, _ = maxs
        x_range = x_max - x_min
        y_range = y_max - y_min
        width = int(np.ceil(x_range / resolution)) + 1
        height = int(np.ceil(y_range / resolution)) + 1

        accumulator, helper = _init_accumulator(statistic, (height, width))

        total_points = reader.header.point_count
        processed = 0

        for chunk_idx, chunk in enumerate(reader.chunk_iterator(chunk_size), start=1):
            xs = np.asarray(chunk.x, dtype=np.float64)
            ys = np.asarray(chunk.y, dtype=np.float64)
            zs = np.asarray(chunk.z, dtype=np.float64)

            xi = np.floor((xs - x_min) / resolution).astype(np.int64)
            yi = np.floor((ys - y_min) / resolution).astype(np.int64)

            valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            if not np.any(valid):
                processed += len(xs)
                continue

            xi = xi[valid]
            yi = yi[valid]
            values = np.asarray(zs[valid], dtype=np.float32)

            if filter_min is not None:
                clip_mask = values >= filter_min
                xi = xi[clip_mask]
                yi = yi[clip_mask]
                values = values[clip_mask]
            if filter_max is not None:
                clip_mask = values <= filter_max
                xi = xi[clip_mask]
                yi = yi[clip_mask]
                values = values[clip_mask]
            if values.size == 0:
                processed += len(xs)
                continue

            _update_accumulator(statistic, accumulator, helper, xi, yi, values)

            processed += len(xs)
            if progress and (chunk_idx % 10 == 0 or processed == total_points):
                percent = (processed / total_points) * 100
                print(
                    f"[depth-map] processed {processed:,}/{total_points:,} "
                    f"points ({percent:.1f}%)",
                )

    depth = _finalize_depth(statistic, accumulator, helper)
    depth_mm = (depth - plane_z) * 1_000.0

    return DepthMapResult(
        depth_mm=depth_mm.astype(np.float32),
        resolution=resolution,
        plane_z=plane_z,
        bbox=(float(x_min), float(y_min), float(x_max), float(y_max)),
        clip_min=filter_min,
        clip_max=filter_max,
        outlier_method=outlier_method,
        iqr_multiplier=iqr_multiplier if outlier_method == "iqr" else None,
        iqr_sample_count=iqr_sample_count,
    )


def save_depthmap_visualization(
    depth_mm: np.ndarray,
    image_path: Path,
    extent: tuple[float, float, float, float] | None = None,
    plane_z: float = 0.0,
    cmap: str = "viridis",
    nan_color: str = "black",
) -> None:
    """Save a false-color visualization of the depth map with a labeled colorbar."""

    if depth_mm.ndim != 2:
        raise ValueError("depth_mm must be a 2D array.")
    if np.all(np.isnan(depth_mm)):
        raise ValueError("depth_mm contains only NaN values; nothing to visualize.")

    image_path = image_path.expanduser()
    image_path.parent.mkdir(parents=True, exist_ok=True)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=nan_color)

    vmin = np.nanmin(depth_mm)
    vmax = np.nanmax(depth_mm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        depth_mm,
        origin="lower",
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Depth from z={plane_z:.3f} m plane (mm)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Orthogonal Depth Map")
    fig.tight_layout()
    fig.savefig(image_path, dpi=300)
    plt.close(fig)
