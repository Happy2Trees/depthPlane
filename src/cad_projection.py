from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


@dataclass
class EdgeOverlay:
    """Outputs of projecting CAD feature edges into the LiDAR map frame."""

    edges_world: np.ndarray  # (E, 2, 3)
    edges_xy: np.ndarray  # (E, 2, 2)
    occupancy: np.ndarray  # (H, W)
    dropped_edges: int
    extent: Tuple[float, float, float, float]
    resolution: float


def load_mesh_vertices(mesh_path: Path, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh and return (vertices, triangles) with optional scaling."""

    mesh_path = mesh_path.expanduser()
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if scale <= 0:
        raise ValueError("scale must be positive")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Mesh at {mesh_path} is empty or unreadable.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.shape[0] == 0 or triangles.shape[0] == 0:
        raise ValueError("Mesh contains no vertices or triangles.")
    return vertices, triangles


def load_transform_matrix(matrix_path: Path) -> np.ndarray:
    """Load a 4x4 homogeneous transform from .txt, .npy, or .npz."""

    matrix_path = matrix_path.expanduser()
    if not matrix_path.exists():
        raise FileNotFoundError(matrix_path)

    if matrix_path.suffix.lower() in {".npy", ".npz"}:
        data = np.load(matrix_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            first_key = next(iter(data.files))
            matrix = data[first_key]
        else:
            matrix = data
    else:
        matrix = np.loadtxt(matrix_path)

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(
            f"Transform must be 4x4 homogeneous; got shape {matrix.shape}.",
        )
    return matrix


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to (N, 3) point cloud rows."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")

    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homogenous = np.concatenate([points, ones], axis=1)
    transformed = homogenous @ transform.T
    return transformed[:, :3]


def _face_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.divide(normals, norms, where=norms > 0)
    return normals


def extract_feature_edges(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    sharp_angle_deg: float = 25.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return boundary + sharp edges as pairs of 3D points."""

    if sharp_angle_deg <= 0 or sharp_angle_deg >= 180:
        raise ValueError("sharp_angle_deg must be in (0, 180)")

    normals = _face_normals(vertices, triangles)
    cos_thresh = np.cos(np.deg2rad(sharp_angle_deg))

    edge_map: dict[tuple[int, int], list[int]] = {}
    for f_idx, tri in enumerate(triangles):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_map.setdefault(key, []).append(f_idx)

    edges: list[tuple[np.ndarray, np.ndarray]] = []
    for (u, v), faces in edge_map.items():
        if len(faces) == 1:
            edges.append((vertices[u], vertices[v]))
        elif len(faces) == 2:
            n1, n2 = normals[faces[0]], normals[faces[1]]
            dot = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
            if dot < cos_thresh:  # sharp
                edges.append((vertices[u], vertices[v]))
        # edges with >2 faces are degenerate; ignore
    return edges


def merge_collinear_edges(
    edges: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    angle_tol_deg: float = 5.0,
    join_tol: float = 0.02,
    min_length: float = 0.02,
    max_passes: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Merge edges that are nearly collinear and share endpoints within tolerance."""

    edges_list = []
    for a, b in edges:
        vec = b - a
        length = float(np.linalg.norm(vec))
        if length < min_length:
            continue
        edges_list.append((a, b))

    if not edges_list:
        return []

    angle_tol_rad = np.deg2rad(angle_tol_deg)
    cos_tol = np.cos(angle_tol_rad)

    def key(pt: np.ndarray) -> tuple[int, int]:
        return tuple(np.round(pt[:2] / join_tol).astype(int))

    for _ in range(max_passes):
        buckets: dict[tuple[int, int], list[int]] = {}
        for idx, (p0, p1) in enumerate(edges_list):
            buckets.setdefault(key(p0), []).append(idx)
            buckets.setdefault(key(p1), []).append(idx)

        used = set()
        new_edges: list[tuple[np.ndarray, np.ndarray]] = []
        merged_any = False

        for i, edge in enumerate(edges_list):
            if i in used:
                continue
            p0, p1 = edge
            dir_edge = p1 - p0
            norm = np.linalg.norm(dir_edge)
            if norm == 0:
                continue
            dir_edge /= norm
            merged = False

            candidates = set(buckets[key(p0)] + buckets[key(p1)])
            candidates.discard(i)

            for j in candidates:
                if j in used:
                    continue
                q0, q1 = edges_list[j]
                shared = None
                for a, name_a in ((p0, "p0"), (p1, "p1")):
                    for b, name_b in ((q0, "q0"), (q1, "q1")):
                        if np.linalg.norm(a[:2] - b[:2]) <= join_tol:
                            shared = (name_a, name_b, a)
                            break
                    if shared:
                        break
                if shared is None:
                    continue

                qdir = q1 - q0
                qnorm = np.linalg.norm(qdir)
                if qnorm == 0:
                    continue
                qdir /= qnorm

                if abs(np.dot(dir_edge, qdir)) < cos_tol:
                    continue

                # Collinear enough; merge endpoints to farthest pair.
                pts = np.array([p0, p1, q0, q1])
                # Project onto dir_edge for ordering.
                proj = pts @ dir_edge
                far_lo = pts[proj.argmin()]
                far_hi = pts[proj.argmax()]
                merged_edge = (far_lo, far_hi)

                used.add(j)
                p0, p1 = merged_edge
                dir_edge = p1 - p0
                norm = np.linalg.norm(dir_edge)
                if norm == 0:
                    break
                dir_edge /= norm
                merged = True
                merged_any = True
            used.add(i)
            new_edges.append((p0, p1))

        edges_list = new_edges
        if not merged_any:
            break

    return edges_list


def filter_axis_aligned_edges(
    edges: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    angle_tol_deg: float = 3.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Keep only edges whose direction is within tolerance of X or Y axis."""

    if angle_tol_deg <= 0 or angle_tol_deg >= 90:
        raise ValueError("angle_tol_deg must be in (0, 90)")
    ang = np.deg2rad(angle_tol_deg)
    cos_tol = np.cos(ang)
    filtered: list[tuple[np.ndarray, np.ndarray]] = []
    for p0, p1 in edges:
        vec = p1 - p0
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        vec /= norm
        vx, vy = float(vec[0]), float(vec[1])
        if abs(vx) >= cos_tol or abs(vy) >= cos_tol:
            filtered.append((p0, p1))
    return filtered


def drop_isolated_edges(
    edges: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    join_tol: float = 0.02,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Remove edges whose both endpoints are isolated (no other edge within tolerance)."""

    edges_list = list(edges)
    if not edges_list:
        return []
    def bucket(pt: np.ndarray) -> tuple[int, int]:
        return tuple(np.round(pt[:2] / join_tol).astype(int))

    counts: dict[tuple[int, int], int] = {}
    for a, b in edges_list:
        counts[bucket(a)] = counts.get(bucket(a), 0) + 1
        counts[bucket(b)] = counts.get(bucket(b), 0) + 1

    kept: list[tuple[np.ndarray, np.ndarray]] = []
    for edge in edges_list:
        a, b = edge
        if counts[bucket(a)] == 1 and counts[bucket(b)] == 1:
            continue
        kept.append(edge)
    return kept


def clip_edges_by_bottom_height(
    edges_world: np.ndarray,
    *,
    mesh_min_z: float,
    clip_bottom_height: float | None,
) -> np.ndarray:
    """Keep only edges whose endpoints are within a bottom slab above mesh_min_z."""

    if clip_bottom_height is None:
        return edges_world
    if clip_bottom_height <= 0:
        raise ValueError("clip_bottom_height must be positive when provided.")
    if edges_world.ndim != 3 or edges_world.shape[1:] != (2, 3):
        raise ValueError("edges_world must have shape (E, 2, 3).")

    z_cut = mesh_min_z + clip_bottom_height
    mask = (edges_world[:, 0, 2] <= z_cut) & (edges_world[:, 1, 2] <= z_cut)
    return edges_world[mask]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _cluster_points(points: np.ndarray, tol: float) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Cluster 2D points within tol using grid buckets + union-find.

    Returns (labels, members_by_label).
    """

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape (N, >=2)")
    if tol <= 0:
        raise ValueError("tol must be positive")

    n = points.shape[0]
    grid = np.floor(points[:, :2] / tol).astype(int)
    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, g in enumerate(grid):
        key = (int(g[0]), int(g[1]))
        buckets.setdefault(key, []).append(idx)

    uf = _UnionFind(n)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for key, idxs in buckets.items():
        gx, gy = key
        candidates = []
        for dx, dy in offsets:
            candidates.extend(buckets.get((gx + dx, gy + dy), []))
        for i in idxs:
            pi = points[i, :2]
            for j in candidates:
                if j <= i:
                    continue
                pj = points[j, :2]
                if np.linalg.norm(pi - pj) <= tol:
                    uf.union(i, j)

    labels = np.array([uf.find(i) for i in range(n)], dtype=int)
    members: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        members.setdefault(lab, []).append(idx)
    return labels, members


def prune_degree_one_edges(
    edges: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    join_tol: float = 0.02,
    max_passes: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Iteratively remove edges whose endpoints are both degree-1 within tolerance."""

    edges_list = list(edges)
    if not edges_list:
        return []

    for _ in range(max_passes):
        pts = []
        for a, b in edges_list:
            pts.append(a)
            pts.append(b)
        pts_arr = np.stack(pts, axis=0)
        labels, _ = _cluster_points(pts_arr, join_tol)

        # degree per cluster
        degree: dict[int, int] = {}
        for ei, (a, b) in enumerate(edges_list):
            la = labels[2 * ei]
            lb = labels[2 * ei + 1]
            if la == lb:
                # collapsed edge; drop it
                degree[la] = degree.get(la, 0)
                continue
            degree[la] = degree.get(la, 0) + 1
            degree[lb] = degree.get(lb, 0) + 1

        kept: list[tuple[np.ndarray, np.ndarray]] = []
        removed_any = False
        for ei, edge in enumerate(edges_list):
            la = labels[2 * ei]
            lb = labels[2 * ei + 1]
            if la == lb:
                removed_any = True
                continue
            deg_a = degree.get(la, 0)
            deg_b = degree.get(lb, 0)
            if deg_a <= 1 and deg_b <= 1:
                removed_any = True
                continue
            kept.append(edge)
        edges_list = kept
        if not removed_any:
            break
    return edges_list


def project_edges_to_grid(
    edges_xy: np.ndarray,
    extent: Tuple[float, float, float, float],
    resolution: float,
    grid_shape: Tuple[int, int],
    line_width_px: int = 1,
) -> tuple[np.ndarray, int]:
    """Rasterize line segments into an occupancy grid; returns (grid, dropped_edges)."""

    if edges_xy.ndim != 3 or edges_xy.shape[1:] != (2, 2):
        raise ValueError("edges_xy must have shape (E, 2, 2)")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if line_width_px <= 0:
        raise ValueError("line_width_px must be positive")

    x_min, x_max, y_min, y_max = extent
    width = grid_shape[1]
    height = grid_shape[0]

    occupancy = np.zeros((height, width), dtype=np.uint32)
    dropped = 0

    for p0, p1 in edges_xy:
        # Convert to pixel coords (float) then sample along the line.
        x0 = (p0[0] - x_min) / resolution
        y0 = (p0[1] - y_min) / resolution
        x1 = (p1[0] - x_min) / resolution
        y1 = (p1[1] - y_min) / resolution

        # Quick reject if completely outside.
        if (
            (x0 < -1 and x1 < -1)
            or (x0 > width and x1 > width)
            or (y0 < -1 and y1 < -1)
            or (y0 > height and y1 > height)
        ):
            dropped += 1
            continue

        dx = x1 - x0
        dy = y1 - y0
        steps = max(int(np.ceil(np.hypot(dx, dy))), 1) + 1
        xs = np.linspace(x0, x1, steps)
        ys = np.linspace(y0, y1, steps)

        xi = np.round(xs).astype(np.int64)
        yi = np.round(ys).astype(np.int64)

        # Apply thickness by offsetting.
        offsets = range(-line_width_px // 2, line_width_px // 2 + 1)
        for ox in offsets:
            for oy in offsets:
                xiu = xi + ox
                yiu = yi + oy
                mask = (xiu >= 0) & (xiu < width) & (yiu >= 0) & (yiu < height)
                if np.any(mask):
                    np.add.at(occupancy, (yiu[mask], xiu[mask]), 1)
    return occupancy, dropped


def project_surface_slice_to_grid(
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    extent: Tuple[float, float, float, float],
    resolution: float,
    grid_shape: Tuple[int, int],
    *,
    plane_z: float,
    slice_mode: str = "top",
    slice_half_thickness: float = 0.03,
    num_samples: int = 200_000,
    include_vertices: bool = True,
    clip_bottom_height: float | None = None,
    rasterize: bool = False,
    subdivide_iters: int = 0,
) -> tuple[np.ndarray, dict[str, int | float | bool | str | None]]:
    """Project a thin surface slice of the CAD mesh into a binary occupancy grid.

    If rasterize=True, optionally subdivide (Loop) then rasterize triangles directly
    onto the grid using a z-buffer style update. Otherwise densify via uniform surface
    sampling (Open3D) and optionally filter sampled points by a bottom slab defined
    by clip_bottom_height (min_z .. min_z + clip_bottom_height).
    For every (x, y) pixel we keep a single candidate along the projection ray:
    - "top": highest z (useful when looking down from +z)
    - "bottom": lowest z
    - "nearest": point whose |z - plane_z| is smallest; optionally restricted by
      `slice_half_thickness` to mimic a cake-slice cross section.
    Pixels with at least one retained candidate are marked 1, others 0.
    """

    if vertices_world.ndim != 2 or vertices_world.shape[1] != 3:
        raise ValueError("vertices_world must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if slice_mode not in {"top", "bottom", "nearest"}:
        raise ValueError("slice_mode must be one of {'top', 'bottom', 'nearest'}")
    if clip_bottom_height is not None and clip_bottom_height <= 0:
        raise ValueError("clip_bottom_height must be positive if provided")
    if subdivide_iters < 0:
        raise ValueError("subdivide_iters must be >= 0")

    x_min, x_max, y_min, y_max = extent
    width = grid_shape[1]
    height = grid_shape[0]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    verts_np = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)

    grid_size = width * height

    if rasterize:
        if subdivide_iters > 0:
            mesh = mesh.subdivide_loop(number_of_iterations=int(subdivide_iters))
            verts_np = np.asarray(mesh.vertices, dtype=np.float64)
            tris = np.asarray(mesh.triangles, dtype=np.int64)

        # Z-buffer style rasterization without point sampling.
        tris_arr = tris
        verts = verts_np

        if slice_mode == "top":
            best = np.full(grid_size, -np.inf, dtype=np.float64)
        elif slice_mode == "bottom":
            best = np.full(grid_size, np.inf, dtype=np.float64)
        else:  # nearest
            best = np.full(grid_size, np.inf, dtype=np.float64)

        for tri_idx in range(tris_arr.shape[0]):
            p_idx = tris_arr[tri_idx]
            p = verts[p_idx]  # (3, 3)
            px, py, pz = p[:, 0], p[:, 1], p[:, 2]

            # Bounding box in pixel coords.
            x0 = (px.min() - x_min) / resolution
            x1 = (px.max() - x_min) / resolution
            y0 = (py.min() - y_min) / resolution
            y1 = (py.max() - y_min) / resolution

            xi_min = max(int(np.floor(x0)), 0)
            xi_max = min(int(np.floor(x1)), width - 1)
            yi_min = max(int(np.floor(y0)), 0)
            yi_max = min(int(np.floor(y1)), height - 1)
            if xi_min > xi_max or yi_min > yi_max:
                continue

            # 2D triangle matrix for barycentric.
            v0 = np.array([px[0], py[0]])
            v1 = np.array([px[1], py[1]])
            v2 = np.array([px[2], py[2]])
            M = np.array([[v1[0] - v0[0], v2[0] - v0[0]], [v1[1] - v0[1], v2[1] - v0[1]]])
            det = np.linalg.det(M)
            if abs(det) < 1e-12:
                continue  # degenerate
            M_inv = np.linalg.inv(M)

            xs = x_min + (np.arange(xi_min, xi_max + 1) + 0.5) * resolution
            ys = y_min + (np.arange(yi_min, yi_max + 1) + 0.5) * resolution
            gx, gy = np.meshgrid(xs, ys)
            q = np.stack([gx - v0[0], gy - v0[1]], axis=-1)  # (..., 2)
            lam = q @ M_inv.T  # (..., 2)
            l1 = lam[..., 0]
            l2 = lam[..., 1]
            l0 = 1.0 - l1 - l2
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not np.any(inside):
                continue

            z_vals = l0 * pz[0] + l1 * pz[1] + l2 * pz[2]

            xi_grid = np.arange(xi_min, xi_max + 1)
            yi_grid = np.arange(yi_min, yi_max + 1)
            xi_full, yi_full = np.meshgrid(xi_grid, yi_grid)
            flat_idx = (yi_full * width + xi_full).ravel()

            mask = inside.ravel()
            flat_idx = flat_idx[mask]
            z_inside = z_vals.ravel()[mask]

            if slice_mode == "nearest":
                z_inside = np.abs(z_inside - plane_z)

            if clip_bottom_height is not None:
                min_z = float(verts[:, 2].min())
                z_cut = min_z + clip_bottom_height
                keep_clip = z_inside <= (z_cut if slice_mode != "nearest" else np.abs(z_cut - plane_z))
                if not np.any(keep_clip):
                    continue
                flat_idx = flat_idx[keep_clip]
                z_inside = z_inside[keep_clip]

            cur = best[flat_idx]
            if slice_mode == "top":
                upd_mask = z_inside > cur
            else:  # bottom or nearest (both use smaller-is-better criteria)
                upd_mask = z_inside < cur

            if np.any(upd_mask):
                best[flat_idx[upd_mask]] = z_inside[upd_mask]

        if slice_mode == "nearest":
            occupancy_flat = best < np.inf
            if slice_half_thickness is not None and slice_half_thickness > 0:
                occupancy_flat &= best <= slice_half_thickness
        else:
            occupancy_flat = best < np.inf if slice_mode == "bottom" else best > -np.inf
        occupancy = occupancy_flat.astype(np.uint8).reshape(height, width)
        sampled = np.empty((0, 3), dtype=np.float64)  # for stats
        points_inside_extent = int(occupancy.sum())
    else:
        # Densify sparse meshes via uniform surface sampling.
        sampled = np.asarray(
            mesh.sample_points_uniformly(number_of_points=int(num_samples)).points,
            dtype=np.float64,
        )
        pts = sampled if not include_vertices else np.concatenate([sampled, verts_np], axis=0)

        if clip_bottom_height is not None:
            min_z = float(verts_np[:, 2].min())
            z_cut = min_z + clip_bottom_height
            keep = pts[:, 2] <= z_cut
            pts = pts[keep]

        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        xi = np.floor((xs - x_min) / resolution).astype(np.int64)
        yi = np.floor((ys - y_min) / resolution).astype(np.int64)
        inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

        xi = xi[inside]
        yi = yi[inside]
        zs = zs[inside]

        points_inside_extent = int(zs.size)

        occupancy_flat: np.ndarray
        flat_idx = yi * width + xi

        if slice_mode == "top":
            best = np.full(grid_size, -np.inf, dtype=np.float64)
            np.maximum.at(best, flat_idx, zs)
            occupancy_flat = best > -np.inf
        elif slice_mode == "bottom":
            best = np.full(grid_size, np.inf, dtype=np.float64)
            np.minimum.at(best, flat_idx, zs)
            occupancy_flat = best < np.inf
        else:  # nearest
            delta = np.abs(zs - plane_z)
            if slice_half_thickness is not None and slice_half_thickness > 0:
                valid = delta <= slice_half_thickness
                delta = delta[valid]
                flat_idx = flat_idx[valid]
            best = np.full(grid_size, np.inf, dtype=np.float64)
            if delta.size:
                np.minimum.at(best, flat_idx, delta)
            occupancy_flat = best < np.inf
            if slice_half_thickness is not None and slice_half_thickness > 0:
                occupancy_flat &= best <= slice_half_thickness

        occupancy = occupancy_flat.astype(np.uint8).reshape(height, width)
        points_inside_extent = int(zs.size)

    stats = {
        "sampled_points": int(sampled.shape[0]),
        "included_vertices": int(include_vertices) * int(verts_np.shape[0]),
        "points_inside_extent": points_inside_extent,
        "occupied_pixels": int(occupancy.sum()),
        "slice_mode": slice_mode,
        "slice_half_thickness": float(slice_half_thickness),
        "clip_bottom_height": None if clip_bottom_height is None else float(clip_bottom_height),
        "rasterize": bool(rasterize),
        "subdivide_iters": int(subdivide_iters),
    }
    return occupancy, stats


def save_projection_images(
    occupancy: np.ndarray,
    extent: Tuple[float, float, float, float],
    projection_image: Path,
    *,
    overlay_base: np.ndarray | None = None,
    overlay_image: Path | None = None,
    plane_z: float = 0.0,
    binarize: bool = True,
) -> None:
    """Save standalone and overlay visualizations of the projected CAD points.

    If binarize=True, visualization uses (occupancy > 0) so counts become a
    clean occupancy mask while still preserving original occupancy array on disk.
    """

    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a 2D array")

    projection_image = projection_image.expanduser()
    projection_image.parent.mkdir(parents=True, exist_ok=True)

    vis = (occupancy > 0).astype(np.uint8) if binarize else occupancy
    is_binary = binarize or vis.dtype == np.bool_ or (
        vis.size > 0 and vis.min() >= 0 and vis.max() <= 1
    )
    if is_binary:
        vmin, vmax = 0.0, 1.0
    else:
        if vis.size:
            vis_min = float(np.nanmin(vis))
            vis_max = float(np.nanmax(vis))
        else:
            vis_min = 0.0
            vis_max = 0.0
        if not np.isfinite(vis_min):
            vis_min = 0.0
        if not np.isfinite(vis_max):
            vis_max = vis_min
        if vis_max == vis_min:
            vis_max = vis_min + 1.0
        vmin, vmax = vis_min, vis_max
    cmap = "binary" if is_binary else "magma"
    cbar_label = "CAD occupancy (1 = filled)" if is_binary else "CAD point hits per pixel"

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        vis,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("CAD Projection")
    fig.tight_layout()
    fig.savefig(projection_image, dpi=300)
    plt.close(fig)

    if overlay_base is not None and overlay_image is not None:
        overlay_image = overlay_image.expanduser()
        overlay_image.parent.mkdir(parents=True, exist_ok=True)

        base_vmin = np.nanmin(overlay_base)
        base_vmax = np.nanmax(overlay_base)

        fig, ax = plt.subplots(figsize=(10, 10))
        base = ax.imshow(
            overlay_base,
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=base_vmin,
            vmax=base_vmax,
            interpolation="nearest",
        )
        cbar = fig.colorbar(base, ax=ax, shrink=0.8)
        cbar.set_label(f"Depth from z={plane_z:.3f} m plane (mm)")

        cad_mask = np.ma.masked_where(vis == 0, vis)
        ax.imshow(
            cad_mask,
            origin="lower",
            extent=extent,
            cmap="autumn" if not is_binary else "Reds",
            alpha=0.45,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Depth Map with CAD Overlay")
        fig.tight_layout()
        fig.savefig(overlay_image, dpi=300)
        plt.close(fig)


def save_projection_mask(occupancy: np.ndarray, mask_image: Path) -> None:
    """Save a pixel-perfect binary mask (black=occupied, white=empty) for segmentation."""

    if occupancy.ndim != 2:
        raise ValueError("occupancy must be a 2D array")

    mask = (occupancy > 0).astype(np.uint8)
    # White background (255), black foreground (0).
    mask_visual = np.where(mask == 1, 0, 255).astype(np.uint8)

    mask_image = mask_image.expanduser()
    mask_image.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(mask_image, mask_visual, cmap="gray", vmin=0, vmax=255)
