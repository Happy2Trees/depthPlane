"""Integration-style check that SAM3 can segment the CAD projection mask.

This variant sprays many random *point* prompts across the drawing to coax SAM3
into segmenting visible parts. It collects the best mask per point, filters tiny
regions, and overlays the kept masks with distinct colors in `tests/results`. The
test skips gracefully when weights or deps are missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = REPO_ROOT / "output" / "cad_projection_mask.png"
RESULTS_DIR = Path(__file__).parent / "results"


def _add_sam3_to_sys_path() -> None:
    sam3_root = REPO_ROOT / "submodules" / "sam3"
    if not sam3_root.exists():
        raise FileNotFoundError("submodules/sam3 is missing; run git submodule update.")
    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))


def _build_point_predictor(device: str):
    """Build SAM3 interactive predictor with backbone, using point prompts."""
    _add_sam3_to_sys_path()
    import torch
    from sam3.model_builder import build_tracker, download_ckpt_from_hf  # type: ignore
    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor  # type: ignore

    checkpoint_path = os.environ.get("SAM3_CKPT_PATH") or download_ckpt_from_hf()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]

    tracker = build_tracker(
        apply_temporal_disambiguation=False, with_backbone=True, compile_mode=None
    )

    tracker_state = {
        k.replace("tracker.", ""): v for k, v in ckpt.items() if k.startswith("tracker.")
    }
    backbone_state = {
        k.replace("detector.backbone.", "backbone."): v
        for k, v in ckpt.items()
        if k.startswith("detector.backbone.")
    }
    tracker_state.update(backbone_state)
    _missing, _unexpected = tracker.load_state_dict(tracker_state, strict=False)

    tracker = tracker.to(device)
    predictor = SAM3InteractiveImagePredictor(tracker).to(device)
    predictor.model.eval()
    return predictor


def _sample_points(image_np: np.ndarray, num_points: int = 64) -> np.ndarray:
    """Pick random foreground-ish pixels to prompt; fallback to uniform sampling."""
    h, w, _ = image_np.shape
    gray = image_np.mean(axis=2)
    fg_mask = gray < 250  # treat near-white as background
    ys, xs = np.where(fg_mask)
    rng = np.random.default_rng(0)
    if len(xs) == 0:
        xs = rng.integers(0, w, size=num_points)
        ys = rng.integers(0, h, size=num_points)
    else:
        idx = rng.choice(len(xs), size=min(num_points, len(xs)), replace=False)
        xs = xs[idx]
        ys = ys[idx]
    return np.stack([xs, ys], axis=1)  # (N,2) in XY pixel coords


def _run_point_prompt_inference(predictor, image_path: Path):
    """Run SAM3 with many point prompts and merge the best mask per point."""
    from PIL import Image

    max_masks = 64
    image = Image.open(image_path).convert("RGB")

    predictor.set_image(image)
    points = _sample_points(np.array(image), num_points=64)

    masks_np = []
    scores_np = []
    for pt in points:
        point_coords = np.array([pt], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)  # foreground
        masks, ious, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=True,
            normalize_coords=False,  # pixel coords
        )
        if masks.size == 0:
            continue
        best_idx = ious.argmax()
        mask = masks[best_idx] > 0.0  # (H, W)
        masks_np.append(mask)
        scores_np.append(float(ious[best_idx]))

    if not masks_np:
        raise AssertionError("SAM3 returned no masks for point prompts.")

    masks = np.stack(masks_np, axis=0)  # (N,H,W)
    scores = np.array(scores_np)

    # Filter out tiny masks
    min_pixels = int(0.0005 * image.width * image.height)
    areas = masks.sum(axis=(1, 2))
    keep = areas >= max(min_pixels, 32)
    masks = masks[keep]
    scores = scores[keep]

    # Limit count
    masks = masks[:max_masks]
    scores = scores[:max_masks]

    if masks.size == 0:
        raise AssertionError("All point-prompt masks were filtered out.")

    return image, masks, scores


def _save_visualizations(
    image,
    masks,
    scores,
    out_dir: Path,
    image_path: Path,
) -> Tuple[Path, Path]:
    """Save overlay PNG and a compact npz with masks/scores."""
    import torch
    from PIL import Image
    from torchvision.utils import draw_segmentation_masks

    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert PIL image -> torch CHW for drawing.
    base = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    palette = [
        (255, 99, 132),
        (99, 200, 255),
        (120, 255, 120),
        (255, 199, 99),
        (199, 99, 255),
        (99, 255, 199),
        (180, 180, 180),
    ]
    colors = [palette[i % len(palette)] for i in range(masks.shape[0])]
    overlay = draw_segmentation_masks(
        base, torch.from_numpy(masks), alpha=0.55, colors=colors
    )
    overlay_path = out_dir / "cad_projection_mask_segments.png"
    overlay_img = overlay.permute(1, 2, 0).byte().cpu().numpy()
    Image.fromarray(overlay_img).save(overlay_path)
    # Save masks + scores for debugging/repro without re-running download.
    data_path = out_dir / "cad_projection_mask_segments.npz"
    np.savez_compressed(
        data_path,
        source_image=str(image_path),
        masks=masks,
        scores=scores,
    )
    return overlay_path, data_path


@pytest.mark.slow
def test_sam3_segments_cad_projection() -> None:
    if not IMAGE_PATH.exists():
        pytest.skip(f"Missing input image: {IMAGE_PATH}")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        pytest.skip(f"torch not available: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        predictor = _build_point_predictor(device=device)
    except Exception as exc:  # pragma: no cover - external download/setup issues
        pytest.skip(f"Skipping SAM3 inference: {exc}")

    image, masks, scores = _run_point_prompt_inference(
        predictor=predictor, image_path=IMAGE_PATH
    )

    # Ensure we actually produced at least one mask.
    assert masks.size > 0, "No segmentation masks generated."

    overlay_path, data_path = _save_visualizations(
        image=image,
        masks=masks,
        scores=scores,
        out_dir=RESULTS_DIR,
        image_path=IMAGE_PATH,
    )

    assert overlay_path.exists(), "Overlay PNG was not written."
    assert data_path.exists(), "NPZ output was not written."
