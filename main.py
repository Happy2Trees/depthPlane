from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import json

import numpy as np
import yaml

from src.cad_projection import (
    apply_transform,
    extract_feature_edges,
    clip_edges_by_bottom_height,
    load_mesh_vertices,
    load_transform_matrix,
    merge_collinear_edges,
    project_edges_to_grid,
    project_surface_slice_to_grid,
    save_projection_images,
    save_projection_mask,
)
from src.depth_map import DepthMapResult, generate_depth_map, save_depthmap_visualization

DEFAULT_CONFIG_PATH = Path("configs/main.yaml")


def _read_yaml_config(path: Path) -> dict[str, Any]:
    """Load YAML mapping; raise if missing or malformed."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def _extract_config_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Translate YAML keys into argparse defaults."""

    def set_if_present(
        source: dict[str, Any],
        key: str,
        dest: str,
        converter,
        defaults: dict[str, Any],
    ) -> None:
        if key in source and source[key] is not None:
            defaults[dest] = converter(source[key])

    defaults: dict[str, Any] = {}

    depth_cfg = config.get("depth_map", {})
    if depth_cfg is not None:
        if not isinstance(depth_cfg, dict):
            raise ValueError("depth_map must be a mapping.")
        set_if_present(depth_cfg, "las", "las", Path, defaults)
        set_if_present(depth_cfg, "resolution_m_per_px", "resolution", float, defaults)
        set_if_present(depth_cfg, "plane_z_m", "plane_z", float, defaults)
        set_if_present(depth_cfg, "statistic", "statistic", str, defaults)
        set_if_present(depth_cfg, "chunk_size", "chunk_size", int, defaults)
        set_if_present(depth_cfg, "clip_min_mm", "clip_min_mm", float, defaults)
        set_if_present(depth_cfg, "clip_max_mm", "clip_max_mm", float, defaults)
        set_if_present(depth_cfg, "output_npy", "output_npy", Path, defaults)
        set_if_present(depth_cfg, "output_image", "output_image", Path, defaults)
        set_if_present(depth_cfg, "meta_json", "meta_json", Path, defaults)

    cad_cfg = config.get("cad_overlay", {})
    if cad_cfg is not None:
        if not isinstance(cad_cfg, dict):
            raise ValueError("cad_overlay must be a mapping.")
        set_if_present(cad_cfg, "mesh", "cad_mesh", Path, defaults)
        set_if_present(cad_cfg, "unit_scale", "cad_unit_scale", float, defaults)
        set_if_present(cad_cfg, "sharp_angle_deg", "cad_sharp_angle_deg", float, defaults)
        set_if_present(cad_cfg, "merge_angle_deg", "cad_merge_angle_deg", float, defaults)
        set_if_present(cad_cfg, "merge_join_m", "cad_merge_join", float, defaults)
        set_if_present(cad_cfg, "min_length_m", "cad_min_length", float, defaults)
        set_if_present(cad_cfg, "line_width_px", "cad_line_width", int, defaults)
        set_if_present(cad_cfg, "transform", "cad_transform", Path, defaults)
        set_if_present(cad_cfg, "points_npy", "cad_points_npy", Path, defaults)
        set_if_present(cad_cfg, "edges_npy", "cad_edges_npy", Path, defaults)
        set_if_present(cad_cfg, "projection_npy", "cad_projection_npy", Path, defaults)
        set_if_present(
            cad_cfg,
            "projection_image",
            "cad_projection_image",
            Path,
            defaults,
        )
        set_if_present(
            cad_cfg,
            "projection_mask_image",
            "cad_projection_mask_image",
            Path,
            defaults,
        )
        set_if_present(cad_cfg, "overlay_image", "cad_overlay_image", Path, defaults)
        set_if_present(cad_cfg, "slice_mode", "cad_slice_mode", str, defaults)
        set_if_present(
            cad_cfg,
            "slice_half_thickness_m",
            "cad_slice_half_thickness",
            float,
            defaults,
        )
        set_if_present(cad_cfg, "sample_mode", "cad_sample_mode", str, defaults)
        set_if_present(cad_cfg, "sample_points", "cad_sample_points", int, defaults)
        set_if_present(
            cad_cfg,
            "clip_bottom_height_m",
            "cad_clip_bottom_height",
            float,
            defaults,
        )
        set_if_present(cad_cfg, "rasterize", "cad_rasterize", bool, defaults)
        set_if_present(cad_cfg, "subdivide_iters", "cad_subdivide_iters", int, defaults)

    return defaults


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an orthogonal depth map (in millimeters) from a LAS point cloud.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to YAML config supplying depth_map/cad_overlay defaults; "
            "CLI flags still override."
        ),
    )
    parser.add_argument(
        "--las",
        type=Path,
        default=Path("data/CAE.las"),
        help="Path to input LAS file (default: data/CAE.las).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.01,
        help="Grid resolution in meters per pixel (default: 0.01 m = 1 cm).",
    )
    parser.add_argument(
        "--plane-z",
        type=float,
        default=0.0,
        help="Reference plane height in meters for depth measurement (default: 0.0).",
    )
    parser.add_argument(
        "--statistic",
        choices=["min", "max", "mean"],
        default="min",
        help="Aggregation for overlapping points (default: min).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2_000_000,
        help="Number of points per chunk while streaming the LAS file (default: 2,000,000).",
    )
    parser.add_argument(
        "--clip-min-mm",
        type=float,
        default=None,
        help="Ignore points with z (m) below this threshold converted from mm (e.g., 1300).",
    )
    parser.add_argument(
        "--clip-max-mm",
        type=float,
        default=None,
        help="Ignore points with z (m) above this threshold converted from mm (e.g., 1600).",
    )
    parser.add_argument(
        "--output-npy",
        type=Path,
        default=Path("output/depth_map.npy"),
        help=(
            "Path to save the depth map in .npy format (default: output/depth_map.npy)."
        ),
    )
    parser.add_argument(
        "--meta-json",
        type=Path,
        default=Path("output/metadata.json"),
        help="Path to save depth-map 메타데이터(JSON) (default: output/metadata.json).",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=Path("visualize/depth_map.png"),
        help="Path to save the depth map visualization (default: visualize/depth_map.png).",
    )
    parser.add_argument(
        "--cad-mesh",
        type=Path,
        default=Path("data/CAD_wo_btm.stl"),
        help="Path to CAD mesh to overlay (default: data/CAD_wo_btm.stl).",
    )
    parser.add_argument(
        "--cad-unit-scale",
        type=float,
        default=0.001,
        help="Scale factor to convert CAD units to meters (use 0.001 for mm -> m).",
    )
    parser.add_argument(
        "--cad-sharp-angle-deg",
        type=float,
        default=25.0,
        help="If face normals differ by more than this, keep the shared edge (deg).",
    )
    parser.add_argument(
        "--cad-merge-angle-deg",
        type=float,
        default=5.0,
        help="Angle tolerance (deg) for merging collinear edges after projection.",
    )
    parser.add_argument(
        "--cad-merge-join",
        type=float,
        default=0.02,
        help="Join tolerance (m) for snapping endpoints while merging edges.",
    )
    parser.add_argument(
        "--cad-min-length",
        type=float,
        default=0.02,
        help="Discard edges shorter than this (m) before merging and rasterizing.",
    )
    parser.add_argument(
        "--cad-line-width",
        type=int,
        default=1,
        help="Line width in pixels when rasterizing edges (default: 1).",
    )
    parser.add_argument(
        "--cad-transform",
        type=Path,
        default=Path("data/pose/final_transformation_matrix.txt"),
        help=(
            "4x4 transform mapping CAD points into LiDAR frame; "
            "applied as-is (no extra inverse)."
        ),
    )
    parser.add_argument(
        "--cad-points-npy",
        type=Path,
        default=Path("output/cad_points_aligned.npy"),
        help=(
            "Where to save transformed CAD sample points "
            "(default: output/cad_points_aligned.npy)."
        ),
    )
    parser.add_argument(
        "--cad-edges-npy",
        type=Path,
        default=Path("output/cad_edges_2d.npy"),
        help="Save 2D projected CAD edges here (default: output/cad_edges_2d.npy).",
    )
    parser.add_argument(
        "--cad-projection-npy",
        type=Path,
        default=Path("output/cad_projection.npy"),
        help=(
            "Where to save the CAD projection grid "
            "(default: output/cad_projection.npy)."
        ),
    )
    parser.add_argument(
        "--cad-projection-image",
        type=Path,
        default=Path("visualize/cad_projection.png"),
        help=(
            "Standalone CAD projection heatmap "
            "(default: visualize/cad_projection.png)."
        ),
    )
    # Optional full-range heatmap is disabled by default; enable with flag.
    parser.add_argument(
        "--save-cad-heatmap",
        action="store_true",
        help="Save an additional full-range CAD occupancy heatmap into output/ (disabled by default).",
    )
    parser.add_argument(
        "--cad-projection-mask-image",
        type=Path,
        default=Path("output/cad_projection_mask.png"),
        help=(
            "Pixel-aligned binary mask (black=occupied, white=empty) for segmentation "
            "(default: output/cad_projection_mask.png)."
        ),
    )
    parser.add_argument(
        "--cad-overlay-image",
        type=Path,
        default=Path("visualize/depth_with_cad.png"),
        help=(
            "Depth-map and CAD overlay visualization "
            "(default: visualize/depth_with_cad.png)."
        ),
    )
    parser.add_argument(
        "--cad-slice-mode",
        choices=["top", "bottom", "nearest"],
        default="top",
        help=(
            "Surface slice to keep when projecting CAD: "
            "top=max z, bottom=min z, nearest=closest to plane_z."
        ),
    )
    parser.add_argument(
        "--cad-slice-half-thickness",
        type=float,
        default=0.03,
        help=(
            "Half thickness (m) around plane_z when slice-mode=nearest; "
            "points farther are ignored."
        ),
    )
    parser.add_argument(
        "--cad-sample-mode",
        choices=["points", "edge"],
        default="points",
        help=(
            "Projection source: 'points' samples CAD surface (existing behavior); "
            "'edge' rasterizes feature edges after clipping to bottom height."
        ),
    )
    parser.add_argument(
        "--cad-sample-points",
        type=int,
        default=200_000,
        help=(
            "Number of surface samples drawn from the CAD mesh before projection "
            "(higher for sparse meshes)."
        ),
    )
    parser.add_argument(
        "--cad-clip-bottom-height",
        type=float,
        default=None,
        help=(
            "If set, pre-clip the CAD mesh to triangles with z within this height above "
            "its minimum z before sampling (meters)."
        ),
    )
    parser.add_argument(
        "--cad-rasterize",
        action="store_true",
        help=(
            "Rasterize triangles directly onto the grid (no point sampling). "
            "Uses z-buffer per pixel respecting slice-mode."
        ),
    )
    parser.add_argument(
        "--cad-subdivide-iters",
        type=int,
        default=0,
        help=(
            "If >0, subdivide CAD triangles (Loop) this many times before clipping/"
            "sampling/rasterizing to densify simplified meshes."
        ),
    )
    parser.add_argument(
        "--no-cad",
        action="store_true",
        help="Skip CAD sampling/projection overlay steps.",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Skip saving the PNG visualization.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence progress logs.",
    )
    preliminary_args, _ = parser.parse_known_args()
    try:
        config_defaults = _extract_config_defaults(
            _read_yaml_config(preliminary_args.config),
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Invalid config file: {exc}", file=sys.stderr)
        sys.exit(1)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    try:
        result: DepthMapResult = generate_depth_map(
            las_path=args.las,
            resolution=args.resolution,
            plane_z=args.plane_z,
            statistic=args.statistic,
            chunk_size=args.chunk_size,
            progress=not args.quiet,
            clip_min=None if args.clip_min_mm is None else args.clip_min_mm / 1000.0,
            clip_max=None if args.clip_max_mm is None else args.clip_max_mm / 1000.0,
        )
    except Exception as exc:  # pragma: no cover - surfaced for CLI users
        print(f"Failed to generate depth map: {exc}", file=sys.stderr)
        raise

    # Depth와 메타데이터 저장
    meta = {
        "resolution_m_per_px": float(result.resolution),
        "plane_z_m": float(result.plane_z),
        "bbox_xy_minmax": tuple(map(float, result.bbox)),
        "extent_xy_minmax": tuple(map(float, result.extent)),
        "statistic": args.statistic,
        "clip_min_mm": None if args.clip_min_mm is None else float(args.clip_min_mm),
        "clip_max_mm": None if args.clip_max_mm is None else float(args.clip_max_mm),
        "depth_shape": tuple(int(x) for x in result.depth_mm.shape),
        "dtype": "float32",
        "description": "Depth map metadata; depth array saved separately in output_npy.",
    }
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, result.depth_mm)
    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    with args.meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"Depth map saved to {args.output_npy} with shape {result.depth_mm.shape} "
        f"and resolution {result.resolution} m/px",
    )
    print(
        f"Depth range: {np.nanmin(result.depth_mm):.1f} mm "
        f"to {np.nanmax(result.depth_mm):.1f} mm",
    )
    print(f"Metadata saved to {args.meta_json}")

    if not args.no_visual:
        save_depthmap_visualization(
            depth_mm=result.depth_mm,
            image_path=args.output_image,
            plane_z=args.plane_z,
            extent=result.extent,
        )
        print(f"Visualization saved to {args.output_image}")

    if not args.no_cad:
        print(
            "\n[CAD overlay] extracting feature edges and projecting "
            f"(mode={args.cad_sample_mode})...",
        )
        verts, tris = load_mesh_vertices(args.cad_mesh, scale=args.cad_unit_scale)
        transform = load_transform_matrix(args.cad_transform)
        # Expect transform already in CAD->LiDAR direction; do not invert here.
        verts_world = apply_transform(verts, transform)
        mesh_min_z = float(verts_world[:, 2].min())

        edges = extract_feature_edges(
            verts_world,
            tris,
            sharp_angle_deg=args.cad_sharp_angle_deg,
        )
        merged_edges = merge_collinear_edges(
            edges,
            angle_tol_deg=args.cad_merge_angle_deg,
            join_tol=args.cad_merge_join,
            min_length=args.cad_min_length,
        )

        edges_world = np.empty((0, 2, 3), dtype=np.float64)
        clip_dropped = 0
        if merged_edges:
            edges_world = np.stack(merged_edges, axis=0)
            if args.cad_sample_mode == "edge" and args.cad_clip_bottom_height is not None:
                before_clip = edges_world.shape[0]
                edges_world = clip_edges_by_bottom_height(
                    edges_world,
                    mesh_min_z=mesh_min_z,
                    clip_bottom_height=args.cad_clip_bottom_height,
                )
                clip_dropped = before_clip - edges_world.shape[0]
        if edges_world.size:
            edges_xy = edges_world[:, :, :2]
        else:
            edges_xy = np.empty((0, 2, 2), dtype=np.float64)

        args.cad_points_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.cad_points_npy, verts_world)

        args.cad_edges_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.cad_edges_npy, edges_xy)

        if edges_xy.size:
            print(
                f"Edges kept: {edges_xy.shape[0]:,} (after merge); "
                f"saved 3D verts to {args.cad_points_npy}, 2D edges to {args.cad_edges_npy}"
            )
        else:
            print(
                "No CAD edges survived filtering; saved aligned vertices only "
                f"to {args.cad_points_npy} and empty edges to {args.cad_edges_npy}"
            )

        occupancy: np.ndarray
        proj_stats: dict[str, int | float | str | None]
        if args.cad_sample_mode == "edge":
            if edges_xy.size:
                occupancy, dropped_extent = project_edges_to_grid(
                    edges_xy,
                    extent=result.extent,
                    resolution=result.resolution,
                    grid_shape=result.depth_mm.shape,
                    line_width_px=args.cad_line_width,
                )
            else:
                occupancy = np.zeros(result.depth_mm.shape, dtype=np.uint32)
                dropped_extent = 0
            proj_stats = {
                "sample_mode": "edge",
                "edges_after_merge": int(edges_xy.shape[0]),
                "dropped_by_bottom_clip": int(clip_dropped),
                "dropped_outside_extent": int(dropped_extent),
                "line_width_px": int(args.cad_line_width),
                "clip_bottom_height": (
                    None
                    if args.cad_clip_bottom_height is None
                    else float(args.cad_clip_bottom_height)
                ),
            }
        else:
            occupancy, proj_stats = project_surface_slice_to_grid(
                verts_world,
                tris,
                extent=result.extent,
                resolution=result.resolution,
                grid_shape=result.depth_mm.shape,
                plane_z=args.plane_z,
                slice_mode=args.cad_slice_mode,
                slice_half_thickness=args.cad_slice_half_thickness,
                num_samples=args.cad_sample_points,
                clip_bottom_height=args.cad_clip_bottom_height,
                rasterize=args.cad_rasterize,
                subdivide_iters=args.cad_subdivide_iters,
            )
            proj_stats["sample_mode"] = "points"

        args.cad_projection_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.cad_projection_npy, occupancy)
        print(f"CAD projection grid saved to {args.cad_projection_npy}")
        print(
            "CAD occupancy range: "
            f"min={np.min(occupancy):.0f}, max={np.max(occupancy):.0f} counts/pixel",
        )
        if args.cad_sample_mode == "edge":
            print(
                "Projection stats: "
                f"edges={proj_stats['edges_after_merge']:,}, "
                f"dropped_clip={proj_stats['dropped_by_bottom_clip']:,}, "
                f"dropped_extent={proj_stats['dropped_outside_extent']:,}, "
                f"line_width_px={proj_stats['line_width_px']}, "
                f"clip_bottom_height={proj_stats['clip_bottom_height']}",
            )
        else:
            print(
                "Projection stats: "
                f"samples={proj_stats['sampled_points']:,}, "
                f"vertices={proj_stats['included_vertices']:,}, "
                f"inside={proj_stats['points_inside_extent']:,}, "
                f"occupied_pixels={proj_stats['occupied_pixels']:,}, "
                f"mode={proj_stats['slice_mode']}, "
                f"thickness={proj_stats['slice_half_thickness']:.3f} m",
            )

        if not args.no_visual:
            # For edge mode, binarize for clearer overlay; keep raw occupancy saved in npy.
            occupancy_vis = (
                (occupancy > 0).astype(np.uint8)
                if args.cad_sample_mode == "edge"
                else occupancy
            )
            save_projection_images(
                occupancy=occupancy_vis,
                extent=result.extent,
                projection_image=args.cad_projection_image,
                overlay_base=result.depth_mm,
                overlay_image=args.cad_overlay_image,
                plane_z=args.plane_z,
                binarize=args.cad_sample_mode == "edge",
            )
            print(f"CAD projection image saved to {args.cad_projection_image}")
            print(f"Depth/CAD overlay image saved to {args.cad_overlay_image}")

            if args.save_cad_heatmap:
                extra_heatmap_path = Path("output/cad_projection_map.png")
                save_projection_images(
                    occupancy=occupancy,
                    extent=result.extent,
                    projection_image=extra_heatmap_path,
                    overlay_base=None,
                    overlay_image=None,
                    plane_z=args.plane_z,
                    binarize=False,
                )
                print(f"CAD occupancy heatmap saved to {extra_heatmap_path}")

        # Always save a pixel-perfect binary mask in output/ for segmentation.
        save_projection_mask(
            occupancy=occupancy,
            mask_image=args.cad_projection_mask_image,
        )
        print(
            "CAD projection mask saved to "
            f"{args.cad_projection_mask_image} (black=occupied, white=empty)",
        )


if __name__ == "__main__":
    main()
