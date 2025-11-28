from __future__ import annotations

"""Quick visual check that CAD pose aligns with the LiDAR point cloud.

Run via Pixi so the pinned Open3D/laspy versions are used::

    pixi run python scripts/visualize_pose_overlay.py \
        --cad data/CAD_wo_btm.stl \
        --las data/CAE.las \
        --transform data/pose/final_transformation_matrix.txt

The script opens an Open3D viewer and optionally saves a top-down scatter PNG.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from src.cad_projection import load_transform_matrix


@dataclass
class LoadedGeometry:
    lidar_points: np.ndarray
    lidar_pcd: o3d.geometry.PointCloud
    cad_vertices: np.ndarray
    cad_mesh: o3d.geometry.TriangleMesh
    cad_wire: o3d.geometry.LineSet


def load_lidar_point_cloud(
    las_path: Path,
    *,
    voxel_size: float = 0.02,
    max_points: int | None = 500_000,
    seed: int = 0,
) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
    """Load a LAS file into an Open3D point cloud with optional downsampling."""

    rng = np.random.default_rng(seed)
    las = laspy.read(str(las_path))
    pts = np.stack([las.x, las.y, las.z], axis=1).astype(np.float64)

    if max_points is not None and pts.shape[0] > max_points:
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.paint_uniform_color([0.55, 0.55, 0.55])
    return pts, pcd


def load_cad_mesh(
    cad_path: Path,
    transform_path: Path,
    *,
    cad_scale: float = 0.001,
) -> tuple[np.ndarray, o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
    """Load, scale, and transform the CAD mesh; return mesh + wireframe."""

    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    if mesh.is_empty():
        raise ValueError(f"CAD mesh at {cad_path} is empty or unreadable.")

    mesh.scale(cad_scale, center=(0.0, 0.0, 0.0))
    transform = load_transform_matrix(transform_path)
    mesh.transform(transform)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.9, 0.2, 0.2])

    wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    wire.paint_uniform_color([1.0, 0.45, 0.1])

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices, mesh, wire


def save_topdown_png(
    lidar_pts: np.ndarray,
    cad_pts: np.ndarray,
    out_path: Path,
    *,
    max_lidar: int = 200_000,
    seed: int = 0,
) -> None:
    """Save a top-down scatter plot comparing LiDAR and transformed CAD."""

    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    lidar_plot = lidar_pts
    if lidar_plot.shape[0] > max_lidar:
        idx = rng.choice(lidar_plot.shape[0], size=max_lidar, replace=False)
        lidar_plot = lidar_plot[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        lidar_plot[:, 0],
        lidar_plot[:, 1],
        s=0.3,
        c="black",
        alpha=0.25,
        linewidths=0,
        label="LiDAR",
    )
    ax.scatter(
        cad_pts[:, 0],
        cad_pts[:, 1],
        s=0.6,
        c="red",
        alpha=0.8,
        linewidths=0,
        label="CAD (transformed)",
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    ax.set_title("Top-down pose check: CAD vs LiDAR")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)



def visualize(geoms: Iterable[o3d.geometry.Geometry], *, window_name: str = "Pose Check") -> None:
    """Open an interactive Open3D window."""

    o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
        list(geoms),
        window_name=window_name,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Visualize CAD pose against LiDAR point cloud.")
    parser.add_argument("--cad", type=Path, default=Path("data/CAD_wo_btm.stl"), help="CAD mesh path")
    parser.add_argument("--las", type=Path, default=Path("data/CAE.las"), help="LiDAR LAS file")
    parser.add_argument(
        "--transform",
        type=Path,
        default=Path("data/pose/final_transformation_matrix.txt"),
        help="4x4 pose that maps CAD into LiDAR frame",
    )
    parser.add_argument("--cad-scale", type=float, default=0.001, help="Scale CAD units to meters (e.g., 0.001 for mm)")
    parser.add_argument("--voxel", type=float, default=0.02, help="Voxel size (m) for LiDAR downsampling")
    parser.add_argument("--max-points", type=int, default=500_000, help="Randomly sample at most this many LiDAR points")
    parser.add_argument("--axis", type=float, default=0.2, help="Coordinate frame size to draw")
    parser.add_argument("--topdown", type=Path, default=Path("visualize/pose_topdown.png"), help="Path to save top-down PNG")
    parser.add_argument("--no-view", action="store_true", help="Skip interactive 3D viewer (only save PNG)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    args = parser.parse_args()

    lidar_pts, lidar_pcd = load_lidar_point_cloud(
        args.las,
        voxel_size=args.voxel,
        max_points=args.max_points,
        seed=args.seed,
    )
    cad_pts, cad_mesh, cad_wire = load_cad_mesh(
        args.cad,
        args.transform,
        cad_scale=args.cad_scale,
    )

    print(
        f"LiDAR points: {lidar_pts.shape[0]:,} (voxel={args.voxel} m, max={args.max_points:,}); "
        f"CAD vertices (after transform): {cad_pts.shape[0]:,}"
    )

    if args.topdown:
        save_topdown_png(lidar_pts, cad_pts, args.topdown, seed=args.seed)
        print(f"Top-down overlay saved to {args.topdown}")

    if not args.no_view:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.axis)
        visualize([lidar_pcd, cad_mesh, cad_wire, axis])


if __name__ == "__main__":
    main()
