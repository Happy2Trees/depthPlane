from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import open3d as o3d

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.GCPnts import GCPnts_AbscissaPoint
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.IGESControl import IGESControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Reader
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, topods
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf


@dataclass
class HLRResult:
    """Outputs of a PythonOCC hidden-line orthographic projection."""

    polylines_world: list[np.ndarray]  # list of (N, 3) arrays in LiDAR frame
    edges_world: np.ndarray  # (E, 2, 3) sampled segments in LiDAR frame
    edges_xy: np.ndarray  # (E, 2, 2) planar segments (x, y only)
    projector_dir: Tuple[float, float, float]


def _read_step_or_iges(shape_path: Path) -> TopoDS_Shape:
    suffix = shape_path.suffix.lower()
    if suffix in {".step", ".stp"}:
        reader = STEPControl_Reader()
    else:
        reader = IGESControl_Reader()
    status = reader.ReadFile(str(shape_path))
    if status != IFSelect_RetDone:
        raise ValueError(f"Failed to read CAD file: {shape_path}")
    reader.TransferRoots()
    return reader.OneShape()


def _stl_to_brep(shape_path: Path) -> TopoDS_Shape:
    """Convert STL triangles into a sewn BRep shell for HLR."""

    mesh = o3d.io.read_triangle_mesh(str(shape_path))
    if mesh.is_empty():
        raise ValueError(f"Mesh at {shape_path} is empty.")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if verts.size == 0 or tris.size == 0:
        raise ValueError("STL has no triangles.")

    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)

    for tri in tris:
        p0, p1, p2 = verts[tri]
        poly = BRepBuilderAPI_MakePolygon()
        poly.Add(gp_Pnt(*map(float, p0)))
        poly.Add(gp_Pnt(*map(float, p1)))
        poly.Add(gp_Pnt(*map(float, p2)))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
        builder.Add(comp, face)

    # Sew faces to unify shared edges; tolerant but fast.
    sewing = BRepBuilderAPI_Sewing()
    sewing.Init(1e-6, True, True, False, True)
    sewing.Load(comp)
    sewing.Perform()
    return sewing.SewedShape()


def load_cad_shape(shape_path: Path) -> TopoDS_Shape:
    """Load STEP/IGES/BREP/STL into a TopoDS_Shape."""

    shape_path = shape_path.expanduser()
    if not shape_path.exists():
        raise FileNotFoundError(shape_path)

    suffix = shape_path.suffix.lower()
    if suffix in {".step", ".stp", ".iges", ".igs"}:
        return _read_step_or_iges(shape_path)
    if suffix == ".brep":
        builder = BRep_Builder()
        shape = TopoDS_Shape()
        if not breptools_Read(shape, str(shape_path), builder):
            raise ValueError(f"Failed to read BREP: {shape_path}")
        return shape
    if suffix == ".stl":
        # STL lacks BRep edges; convert to sewn faces for HLR.
        return _stl_to_brep(shape_path)

    raise ValueError(
        f"Unsupported CAD format '{suffix}'. Use STEP/IGES/BREP/STL for HLR.",
    )


def _matrix_to_trsf(matrix: np.ndarray) -> gp_Trsf:
    """Convert a 4x4 homogeneous matrix to gp_Trsf (rotation + translation)."""

    if matrix.shape != (4, 4):
        raise ValueError("transform matrix must be 4x4.")

    r = matrix[:3, :3]
    t = matrix[:3, 3]
    trsf = gp_Trsf()
    trsf.SetValues(
        float(r[0, 0]),
        float(r[0, 1]),
        float(r[0, 2]),
        float(t[0]),
        float(r[1, 0]),
        float(r[1, 1]),
        float(r[1, 2]),
        float(t[1]),
        float(r[2, 0]),
        float(r[2, 1]),
        float(r[2, 2]),
        float(t[2]),
    )
    return trsf


def transform_shape(
    shape: TopoDS_Shape,
    *,
    unit_scale: float,
    transform_matrix: np.ndarray,
) -> TopoDS_Shape:
    """Apply uniform scale then a rigid pose to a shape."""

    if unit_scale <= 0:
        raise ValueError("unit_scale must be positive.")

    scaled = shape
    if not np.isclose(unit_scale, 1.0):
        scale_trsf = gp_Trsf()
        scale_trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), float(unit_scale))
        scaled = BRepBuilderAPI_Transform(scaled, scale_trsf, True).Shape()

    pose_trsf = _matrix_to_trsf(transform_matrix)
    return BRepBuilderAPI_Transform(scaled, pose_trsf, True).Shape()


def _edge_to_polyline(edge, *, step: float) -> np.ndarray | None:
    """Sample a TopoDS_Edge to a polyline; fall back to endpoints if length is ill-defined."""

    adaptor = BRepAdaptor_Curve(edge)
    first = adaptor.FirstParameter()
    last = adaptor.LastParameter()
    length = None
    try:
        length = GCPnts_AbscissaPoint.Length(adaptor, first, last)
    except Exception:
        length = None

    if length is None or not np.isfinite(length) or length <= 0:
        n_pts = 2
    else:
        n_pts = max(2, int(np.ceil(length / step)))

    params = np.linspace(first, last, n_pts)
    pts: List[np.ndarray] = []
    for u in params:
        pnt = adaptor.Value(float(u))
        pts.append(np.array([pnt.X(), pnt.Y(), pnt.Z()], dtype=np.float64))
    if len(pts) < 2:
        return None
    return np.stack(pts, axis=0)


def _shape_to_polylines(
    shape: TopoDS_Shape,
    *,
    step: float,
) -> list[np.ndarray]:
    """Discretize all edges of a shape into polylines."""

    polylines: list[np.ndarray] = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = topods.Edge(explorer.Current())
        poly = _edge_to_polyline(edge, step=step)
        if poly is not None:
            polylines.append(poly)
        explorer.Next()
    return polylines


def _polylines_to_segments(
    polylines: Iterable[np.ndarray],
    *,
    min_length: float,
) -> list[np.ndarray]:
    """Convert polylines to line segments while dropping very short pieces."""

    segments: list[np.ndarray] = []
    for poly in polylines:
        if poly.shape[0] < 2:
            continue
        for i in range(poly.shape[0] - 1):
            p0 = poly[i]
            p1 = poly[i + 1]
            if np.linalg.norm(p1 - p0) < min_length:
                continue
            segments.append(np.stack([p0, p1], axis=0))
    return segments


def run_hlr_projection(
    cad_path: Path,
    *,
    unit_scale: float,
    transform_matrix: np.ndarray,
    projection_dir: Sequence[float] = (0.0, 0.0, 1.0),
    sample_step: float = 0.01,
    min_length: float = 0.0,
) -> HLRResult:
    """Load CAD with PythonOCC, apply pose, run HLR, and sample visible edges."""

    if sample_step <= 0:
        raise ValueError("sample_step must be positive.")
    if min_length < 0:
        raise ValueError("min_length must be non-negative.")

    shape = load_cad_shape(cad_path)
    posed_shape = transform_shape(
        shape,
        unit_scale=unit_scale,
        transform_matrix=transform_matrix,
    )

    dir_vec = np.array(projection_dir, dtype=np.float64).flatten()
    if dir_vec.shape[0] != 3 or np.allclose(dir_vec, 0):
        raise ValueError("projection_dir must be a 3-vector and non-zero.")
    dir_norm = dir_vec / np.linalg.norm(dir_vec)
    projector = HLRAlgo_Projector(
        gp_Ax2(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(*dir_norm.tolist())),
    )

    algo = HLRBRep_Algo()
    algo.Add(posed_shape)
    algo.Projector(projector)
    algo.Update()
    algo.Hide()

    hlr_shapes = HLRBRep_HLRToShape(algo)
    visible = hlr_shapes.VCompound()

    polylines = _shape_to_polylines(visible, step=sample_step)
    segments = _polylines_to_segments(polylines, min_length=min_length)

    if segments:
        edges_world = np.stack(segments, axis=0)
        edges_xy = edges_world[:, :, :2]
    else:
        edges_world = np.empty((0, 2, 3), dtype=np.float64)
        edges_xy = np.empty((0, 2, 2), dtype=np.float64)

    return HLRResult(
        polylines_world=polylines,
        edges_world=edges_world,
        edges_xy=edges_xy,
        projector_dir=(float(dir_norm[0]), float(dir_norm[1]), float(dir_norm[2])),
    )
