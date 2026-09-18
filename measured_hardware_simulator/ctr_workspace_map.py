"""Generate and load sampled reachable-tip caches for the CTR viewer.

These six-input XYZ caches support interaction, target projection and solver
restarts. They are not protocol-v1 statistical analysis datasets; formal design
analysis uses the five-DOF canonical banks in :mod:`ctr_sampling`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from vispy.visuals.isosurface import isosurface

from CTR_superPosKin_fun_sectioned import superPosKin


DEFAULT_SAMPLE_COUNT = 12_000
DEFAULT_SEED = 42
DEFAULT_WORKSPACE_MAP_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "workspace"
    / "ctr_reachable_workspace_12000.npz"
)
DEFAULT_ZONE_SAMPLE_COUNT = 80_000
DEFAULT_REACHABILITY_ZONES_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "workspace"
    / "ctr_reachability_zones_80000.npz"
)
DEFAULT_ENDPOINT_WORKSPACE_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "workspace"
    / "ctr_endpoint_workspaces_12000.npz"
)


@dataclass(frozen=True)
class WorkspaceMap:
    """Sampled tip positions and the valid tube states that produced them."""

    tips_mm: np.ndarray
    deployment_m: np.ndarray
    rotation_rad: np.ndarray


@dataclass(frozen=True)
class EndpointWorkspaceMaps:
    """Reachable positions for all three tube endpoints at shared states."""

    tips_mm: np.ndarray
    deployment_m: np.ndarray
    rotation_rad: np.ndarray

    def for_tube(self, tube: int) -> WorkspaceMap:
        tube_index = int(tube)
        if tube_index not in (0, 1, 2):
            raise ValueError("tube must be 0 (inner), 1 (middle), or 2 (outer)")
        return WorkspaceMap(
            tips_mm=self.tips_mm[tube_index],
            deployment_m=self.deployment_m,
            rotation_rad=self.rotation_rad,
        )


@dataclass(frozen=True)
class WorkspaceSurface:
    """A smooth surface-of-revolution approximation of the workspace boundary."""

    vertices_mm: np.ndarray
    faces: np.ndarray
    profile_z_mm: np.ndarray
    profile_radius_mm: np.ndarray


@dataclass(frozen=True)
class ZoneMesh:
    """One triangulated reachability classification boundary."""

    vertices_mm: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class ReachabilityZones:
    """Cached blue/red/grey workspace-zone meshes and their metadata."""

    reachable: ZoneMesh
    uncertain: ZoneMesh
    inaccessible: ZoneMesh
    sample_count: int
    voxel_size_mm: float
    reachable_distance_mm: float
    uncertain_distance_mm: float
    cutaway_quadrant: bool
    reachable_voxels: int
    uncertain_voxels: int
    inaccessible_voxels: int


def sample_valid_configurations(
    total_lengths_m: np.ndarray,
    sample_count: int,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample nested deployments and full-revolution rotations for the viewer.

    Deployments span the complete valid range, rather than holding any tube at
    full extension.  A small set of exact boundary samples is included so the
    map represents both fully retracted and fully extended configurations. The
    injected boundary rows and redundant common rotation make this unsuitable
    for protocol-v1 independent statistical counts.
    """
    lengths = np.asarray(total_lengths_m, dtype=float).reshape(-1)
    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        raise ValueError("total_lengths_m must contain three positive lengths")
    if not lengths[0] >= lengths[1] >= lengths[2]:
        raise ValueError("tube lengths must be ordered inner >= middle >= outer")
    if sample_count < 8:
        raise ValueError("sample_count must be at least 8")

    rng = np.random.default_rng(seed)
    fractions = rng.random((sample_count, 3))

    # The fraction cube maps to valid nested deployments. Include its eight
    # corners exactly so the sampled cloud contains its physical boundaries.
    fractions[:8] = np.asarray(
        [
            [inner, middle, outer]
            for inner in (0.0, 1.0)
            for middle in (0.0, 1.0)
            for outer in (0.0, 1.0)
        ],
        dtype=float,
    )

    outer = fractions[:, 2] * lengths[2]
    middle = outer + fractions[:, 1] * (lengths[1] - outer)
    inner = middle + fractions[:, 0] * (lengths[0] - middle)
    deployment = np.column_stack((inner, middle, outer))
    rotation = rng.uniform(-np.pi, np.pi, size=(sample_count, 3))
    return deployment, rotation


def generate_workspace_map(
    parameters: dict,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    seed: int = DEFAULT_SEED,
    model_points_per_section: int = 2,
) -> WorkspaceMap:
    """Evaluate a deterministic sample of the full six-input workspace."""
    total_lengths = np.asarray(
        [sum(lengths) for lengths in parameters["l_t"]], dtype=float
    )
    deployment, rotation = sample_valid_configurations(
        total_lengths,
        sample_count,
        seed,
    )
    tips_mm = np.empty((sample_count, 3), dtype=np.float32)
    simulation = {"n_p": int(model_points_per_section), "isPlot": False}

    for index in range(sample_count):
        result = superPosKin(
            parameters,
            {
                "ul": deployment[index].tolist(),
                "uphi": rotation[index].tolist(),
            },
            simulation,
        )
        tip_mm = np.asarray(result[0][:3], dtype=float) * 1000.0
        if not np.all(np.isfinite(tip_mm)):
            raise RuntimeError(f"non-finite workspace point at sample {index}")
        tips_mm[index] = tip_mm

    return WorkspaceMap(
        tips_mm=tips_mm,
        deployment_m=deployment.astype(np.float32),
        rotation_rad=rotation.astype(np.float32),
    )


def save_workspace_map(
    workspace: WorkspaceMap,
    path: Path = DEFAULT_WORKSPACE_MAP_PATH,
) -> None:
    """Save a compressed map that can be loaded quickly by the live viewer."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        tips_mm=np.asarray(workspace.tips_mm, dtype=np.float32),
        deployment_m=np.asarray(workspace.deployment_m, dtype=np.float32),
        rotation_rad=np.asarray(workspace.rotation_rad, dtype=np.float32),
    )


def load_workspace_map(
    path: Path = DEFAULT_WORKSPACE_MAP_PATH,
) -> WorkspaceMap:
    """Load and validate the precomputed workspace map."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as stored:
        tips_mm = np.asarray(stored["tips_mm"], dtype=np.float32)
        deployment_m = np.asarray(stored["deployment_m"], dtype=np.float32)
        rotation_rad = np.asarray(stored["rotation_rad"], dtype=np.float32)

    if (
        tips_mm.ndim != 2
        or tips_mm.shape[1] != 3
        or deployment_m.shape != tips_mm.shape
        or rotation_rad.shape != tips_mm.shape
        or not np.all(np.isfinite(tips_mm))
        or not np.all(np.isfinite(deployment_m))
        or not np.all(np.isfinite(rotation_rad))
    ):
        raise ValueError(f"Invalid CTR workspace map: {source}")

    return WorkspaceMap(tips_mm, deployment_m, rotation_rad)


def generate_endpoint_workspace_maps(
    parameters: dict,
    workspace: WorkspaceMap,
    *,
    model_points_per_section: int = 2,
) -> EndpointWorkspaceMaps:
    """Evaluate inner, middle and outer endpoints at shared sampled states."""
    # Imported lazily to keep the basic workspace generator independent of the
    # inverse-kinematics module at import time.
    from ctr_inverse_kinematics import ConstrainedTipIK

    solver = ConstrainedTipIK(
        parameters,
        model_points_per_section=model_points_per_section,
    )
    sample_count = len(workspace.tips_mm)
    endpoint_tips = np.empty((3, sample_count, 3), dtype=np.float32)
    for index in range(sample_count):
        endpoint_tips[:, index] = solver.forward_endpoints_mm(
            workspace.deployment_m[index],
            workspace.rotation_rad[index],
        )
    return EndpointWorkspaceMaps(
        tips_mm=endpoint_tips,
        deployment_m=np.asarray(workspace.deployment_m, dtype=np.float32),
        rotation_rad=np.asarray(workspace.rotation_rad, dtype=np.float32),
    )


def save_endpoint_workspace_maps(
    workspace: EndpointWorkspaceMaps,
    path: Path = DEFAULT_ENDPOINT_WORKSPACE_PATH,
) -> None:
    """Store endpoint-specific workspace samples for fast viewer startup."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        tips_mm=np.asarray(workspace.tips_mm, dtype=np.float32),
        deployment_m=np.asarray(workspace.deployment_m, dtype=np.float32),
        rotation_rad=np.asarray(workspace.rotation_rad, dtype=np.float32),
    )


def load_endpoint_workspace_maps(
    path: Path = DEFAULT_ENDPOINT_WORKSPACE_PATH,
) -> EndpointWorkspaceMaps:
    """Load and validate endpoint-specific sampled workspaces."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as stored:
        tips_mm = np.asarray(stored["tips_mm"], dtype=np.float32)
        deployment_m = np.asarray(stored["deployment_m"], dtype=np.float32)
        rotation_rad = np.asarray(stored["rotation_rad"], dtype=np.float32)
    if (
        tips_mm.ndim != 3
        or tips_mm.shape[0] != 3
        or tips_mm.shape[2] != 3
        or deployment_m.shape != tips_mm.shape[1:]
        or rotation_rad.shape != tips_mm.shape[1:]
        or not np.all(np.isfinite(tips_mm))
        or not np.all(np.isfinite(deployment_m))
        or not np.all(np.isfinite(rotation_rad))
    ):
        raise ValueError(f"Invalid endpoint workspace maps: {source}")
    return EndpointWorkspaceMaps(tips_mm, deployment_m, rotation_rad)


def radial_workspace_envelope(
    tips_mm: np.ndarray,
    *,
    height_sections: int = 72,
    boundary_quantile: float = 0.995,
    smoothing_passes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a smooth radial boundary as a function of global Z.

    The unrestricted tube rotations make the sampled workspace approximately
    axisymmetric. Every angular direction is retained before XYZ samples are
    collapsed to cylindrical ``(radius, Z)`` coordinates, so the envelope does
    not rely on a single quadrant being perfectly representative.
    """
    points = np.asarray(tips_mm, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < height_sections
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("tips_mm must contain finite XYZ workspace samples")
    if height_sections < 8:
        raise ValueError("height_sections must be at least 8")
    if not 0.9 <= boundary_quantile <= 1.0:
        raise ValueError("boundary_quantile must be between 0.9 and 1.0")
    if smoothing_passes < 0:
        raise ValueError("smoothing_passes cannot be negative")

    radius = np.hypot(points[:, 0], points[:, 1])
    height = points[:, 2]
    z_min = float(np.min(height))
    z_max = float(np.max(height))
    if z_max - z_min <= 1e-9:
        raise ValueError("workspace samples must span more than one Z value")

    edges = np.linspace(z_min, z_max, height_sections + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    band = np.clip(np.digitize(height, edges) - 1, 0, height_sections - 1)
    envelope = np.full(height_sections, np.nan, dtype=float)
    for index in range(height_sections):
        values = radius[band == index]
        if len(values):
            envelope[index] = float(np.quantile(values, boundary_quantile))

    available = np.isfinite(envelope)
    if np.count_nonzero(available) < 2:
        raise ValueError("workspace samples do not cover enough height bands")
    envelope[~available] = np.interp(
        centres[~available],
        centres[available],
        envelope[available],
    )

    # A short binomial filter removes sampling spikes without inventing a
    # high-order polynomial profile. Endpoints are added as physical poles
    # after smoothing so the shell closes cleanly at the base and maximum reach.
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    for _ in range(smoothing_passes):
        padded = np.pad(envelope, (2, 2), mode="edge")
        envelope = np.convolve(padded, kernel, mode="valid")

    profile_z = np.concatenate(([z_min], centres, [z_max]))
    profile_radius = np.concatenate(
        ([0.0], np.maximum(envelope, 0.0), [0.0])
    )
    return profile_z.astype(np.float32), profile_radius.astype(np.float32)


def workspace_surface_of_revolution(
    tips_mm: np.ndarray,
    *,
    height_sections: int = 72,
    azimuth_sections: int = 120,
    boundary_quantile: float = 0.995,
    smoothing_passes: int = 3,
) -> WorkspaceSurface:
    """Revolve the radial workspace envelope into a closed 360-degree mesh."""
    if azimuth_sections < 12:
        raise ValueError("azimuth_sections must be at least 12")
    profile_z, profile_radius = radial_workspace_envelope(
        tips_mm,
        height_sections=height_sections,
        boundary_quantile=boundary_quantile,
        smoothing_passes=smoothing_passes,
    )

    angles = np.linspace(0.0, 2.0 * np.pi, azimuth_sections, endpoint=False)
    ring_z = profile_z[1:-1]
    ring_radius = profile_radius[1:-1]
    ring_vertices = np.asarray(
        [
            [radius * np.cos(angle), radius * np.sin(angle), height]
            for height, radius in zip(ring_z, ring_radius)
            for angle in angles
        ],
        dtype=np.float32,
    )
    vertices = np.vstack(
        (
            np.asarray([[0.0, 0.0, profile_z[0]]], dtype=np.float32),
            ring_vertices,
            np.asarray([[0.0, 0.0, profile_z[-1]]], dtype=np.float32),
        )
    )

    faces: list[tuple[int, int, int]] = []
    first_ring = 1
    for angle_index in range(azimuth_sections):
        following = (angle_index + 1) % azimuth_sections
        faces.append((0, first_ring + following, first_ring + angle_index))

    ring_count = len(ring_z)
    for ring_index in range(ring_count - 1):
        lower = first_ring + ring_index * azimuth_sections
        upper = lower + azimuth_sections
        for angle_index in range(azimuth_sections):
            following = (angle_index + 1) % azimuth_sections
            faces.append((lower + angle_index, lower + following, upper + angle_index))
            faces.append((lower + following, upper + following, upper + angle_index))

    top = len(vertices) - 1
    last_ring = first_ring + (ring_count - 1) * azimuth_sections
    for angle_index in range(azimuth_sections):
        following = (angle_index + 1) % azimuth_sections
        faces.append((last_ring + angle_index, last_ring + following, top))

    return WorkspaceSurface(
        vertices_mm=vertices,
        faces=np.asarray(faces, dtype=np.uint32),
        profile_z_mm=profile_z,
        profile_radius_mm=profile_radius,
    )


def constrain_point_to_workspace_surface(
    point_mm: np.ndarray,
    profile_z_mm: np.ndarray,
    profile_radius_mm: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Project an XYZ point into a smooth axisymmetric workspace envelope.

    The radial/Z profile is treated as a closed piecewise-linear region. A
    point already inside it is returned unchanged; an outside point is moved
    to the closest profile or axis segment so target motion slides along the
    boundary instead of jumping through it.
    """
    point = np.asarray(point_mm, dtype=float).reshape(3)
    profile_z = np.asarray(profile_z_mm, dtype=float).reshape(-1)
    profile_radius = np.asarray(profile_radius_mm, dtype=float).reshape(-1)
    if (
        len(profile_z) < 2
        or profile_radius.shape != profile_z.shape
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(profile_z))
        or not np.all(np.isfinite(profile_radius))
        or np.any(np.diff(profile_z) < 0.0)
        or np.any(profile_radius < 0.0)
    ):
        raise ValueError("workspace profile and point must be finite and valid")

    radial = float(np.hypot(point[0], point[1]))
    height = float(point[2])
    if profile_z[0] <= height <= profile_z[-1]:
        allowed_radius = float(np.interp(height, profile_z, profile_radius))
        if radial <= allowed_radius + 1e-9:
            return point.copy(), False

    polygon = np.column_stack((profile_radius, profile_z))
    polygon = np.vstack((polygon, polygon[0]))
    query = np.asarray([radial, height], dtype=float)
    best = polygon[0]
    best_distance_sq = float("inf")
    for start, end in zip(polygon[:-1], polygon[1:]):
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-18:
            candidate = start
        else:
            fraction = float(
                np.clip(np.dot(query - start, segment) / length_sq, 0.0, 1.0)
            )
            candidate = start + fraction * segment
        distance_sq = float(np.dot(query - candidate, query - candidate))
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best = candidate

    if radial > 1e-12:
        radial_scale = float(best[0]) / radial
        projected_xy = point[:2] * radial_scale
    else:
        projected_xy = np.zeros(2, dtype=float)
    projected = np.asarray([projected_xy[0], projected_xy[1], best[1]])
    return projected, True


def _zone_mesh(
    mask: np.ndarray,
    origin_mm: np.ndarray,
    voxel_size_mm: float,
    smoothing_sigma_voxels: float,
) -> ZoneMesh:
    """Convert one voxel mask into a closed, lightly smoothed triangle mesh."""
    field = gaussian_filter(
        np.asarray(mask, dtype=np.float32),
        sigma=float(smoothing_sigma_voxels),
        mode="constant",
    )
    field = np.pad(field, 1, mode="constant")
    if np.max(field) < 0.5:
        return ZoneMesh(
            vertices_mm=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.uint32),
        )
    with warnings.catch_warnings():
        # VisPy's marching-cubes lookup uses unsigned integer arithmetic and
        # emits harmless overflow warnings on recent NumPy versions.
        warnings.simplefilter("ignore", RuntimeWarning)
        vertices, faces = isosurface(field, 0.5)
    vertices_mm = (
        np.asarray(origin_mm, dtype=np.float32)
        + (np.asarray(vertices, dtype=np.float32) - 1.0) * float(voxel_size_mm)
    )
    return ZoneMesh(
        vertices_mm=vertices_mm,
        faces=np.asarray(faces, dtype=np.uint32),
    )


def generate_reachability_zones(
    workspace: WorkspaceMap,
    *,
    voxel_size_mm: float = 8.0,
    reachable_distance_mm: float = 10.0,
    uncertain_distance_mm: float = 18.0,
    smoothing_sigma_voxels: float = 1.15,
    cutaway_quadrant: bool = False,
) -> ReachabilityZones:
    """Classify a finite workspace envelope into three estimated zones.

    Blue/reachable voxels lie close to a forward-kinematics sample. Grey voxels
    form a buffer where sampling is inconclusive. Red/inaccessible voxels are
    farther from every sampled solution but still inside the estimated outer
    envelope. Consequently, red means "no solution found at this resolution",
    not a mathematical proof that no actuator state can reach the voxel.
    """
    tips = np.asarray(workspace.tips_mm, dtype=float)
    if tips.ndim != 2 or tips.shape[1] != 3 or not np.all(np.isfinite(tips)):
        raise ValueError("workspace tips must be a finite N-by-3 array")
    if voxel_size_mm <= 0.0:
        raise ValueError("voxel_size_mm must be positive")
    if not 0.0 < reachable_distance_mm < uncertain_distance_mm:
        raise ValueError(
            "distance thresholds must satisfy 0 < reachable < uncertain"
        )

    profile_z, profile_radius = radial_workspace_envelope(
        tips,
        height_sections=72,
        boundary_quantile=0.999,
        smoothing_passes=2,
    )
    maximum_radius = float(np.ceil(np.max(profile_radius) / voxel_size_mm))
    maximum_radius *= voxel_size_mm
    z_min = float(np.floor(np.min(profile_z) / voxel_size_mm) * voxel_size_mm)
    z_max = float(np.ceil(np.max(profile_z) / voxel_size_mm) * voxel_size_mm)

    x_values = np.arange(
        -maximum_radius,
        maximum_radius + voxel_size_mm * 0.5,
        voxel_size_mm,
        dtype=np.float32,
    )
    y_values = x_values.copy()
    z_values = np.arange(
        z_min,
        z_max + voxel_size_mm * 0.5,
        voxel_size_mm,
        dtype=np.float32,
    )
    x_grid, y_grid, z_grid = np.meshgrid(
        x_values,
        y_values,
        z_values,
        indexing="ij",
    )
    query_points = np.column_stack(
        (x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
    )
    nearest_distance = cKDTree(tips).query(query_points, workers=-1)[0]
    nearest_distance = nearest_distance.reshape(x_grid.shape)

    radius_grid = np.hypot(x_grid, y_grid)
    allowed_radius = np.interp(
        z_grid,
        profile_z,
        profile_radius,
        left=0.0,
        right=0.0,
    )
    analysis_domain = radius_grid <= allowed_radius + voxel_size_mm * 0.5
    if cutaway_quadrant:
        # Optional diagnostic cutaway; the live viewer uses the complete dome.
        analysis_domain &= ~((x_grid > 0.0) & (y_grid > 0.0))

    reachable = analysis_domain & (nearest_distance <= reachable_distance_mm)
    uncertain = analysis_domain & (
        (nearest_distance > reachable_distance_mm)
        & (nearest_distance <= uncertain_distance_mm)
    )
    inaccessible = analysis_domain & (nearest_distance > uncertain_distance_mm)
    origin = np.asarray([x_values[0], y_values[0], z_values[0]], dtype=np.float32)

    return ReachabilityZones(
        reachable=_zone_mesh(
            reachable,
            origin,
            voxel_size_mm,
            smoothing_sigma_voxels,
        ),
        uncertain=_zone_mesh(
            uncertain,
            origin,
            voxel_size_mm,
            min(smoothing_sigma_voxels, 0.9),
        ),
        inaccessible=_zone_mesh(
            inaccessible,
            origin,
            voxel_size_mm,
            min(smoothing_sigma_voxels, 0.75),
        ),
        sample_count=len(tips),
        voxel_size_mm=float(voxel_size_mm),
        reachable_distance_mm=float(reachable_distance_mm),
        uncertain_distance_mm=float(uncertain_distance_mm),
        cutaway_quadrant=bool(cutaway_quadrant),
        reachable_voxels=int(np.count_nonzero(reachable)),
        uncertain_voxels=int(np.count_nonzero(uncertain)),
        inaccessible_voxels=int(np.count_nonzero(inaccessible)),
    )


def save_reachability_zones(
    zones: ReachabilityZones,
    path: Path = DEFAULT_REACHABILITY_ZONES_PATH,
) -> None:
    """Save prebuilt meshes so the live viewer starts without voxel analysis."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        reachable_vertices_mm=zones.reachable.vertices_mm,
        reachable_faces=zones.reachable.faces,
        uncertain_vertices_mm=zones.uncertain.vertices_mm,
        uncertain_faces=zones.uncertain.faces,
        inaccessible_vertices_mm=zones.inaccessible.vertices_mm,
        inaccessible_faces=zones.inaccessible.faces,
        metadata=np.asarray(
            [
                zones.sample_count,
                zones.voxel_size_mm,
                zones.reachable_distance_mm,
                zones.uncertain_distance_mm,
                zones.reachable_voxels,
                zones.uncertain_voxels,
                zones.inaccessible_voxels,
                int(zones.cutaway_quadrant),
            ],
            dtype=np.float64,
        ),
    )


def load_reachability_zones(
    path: Path = DEFAULT_REACHABILITY_ZONES_PATH,
) -> ReachabilityZones:
    """Load the cached reachability-zone meshes used by the live viewer."""
    with np.load(Path(path), allow_pickle=False) as stored:
        reachable = ZoneMesh(
            np.asarray(stored["reachable_vertices_mm"], dtype=np.float32),
            np.asarray(stored["reachable_faces"], dtype=np.uint32),
        )
        uncertain = ZoneMesh(
            np.asarray(stored["uncertain_vertices_mm"], dtype=np.float32),
            np.asarray(stored["uncertain_faces"], dtype=np.uint32),
        )
        inaccessible = ZoneMesh(
            np.asarray(stored["inaccessible_vertices_mm"], dtype=np.float32),
            np.asarray(stored["inaccessible_faces"], dtype=np.uint32),
        )
        metadata = np.asarray(stored["metadata"], dtype=float)

    for name, mesh in (
        ("reachable", reachable),
        ("uncertain", uncertain),
        ("inaccessible", inaccessible),
    ):
        if (
            mesh.vertices_mm.ndim != 2
            or mesh.vertices_mm.shape[1] != 3
            or mesh.faces.ndim != 2
            or mesh.faces.shape[1] != 3
            or not np.all(np.isfinite(mesh.vertices_mm))
            or (len(mesh.faces) and np.max(mesh.faces) >= len(mesh.vertices_mm))
        ):
            raise ValueError(f"Invalid {name} mesh in {path}")
    if metadata.shape != (8,) or not np.all(np.isfinite(metadata)):
        raise ValueError(f"Invalid reachability metadata in {path}")

    return ReachabilityZones(
        reachable=reachable,
        uncertain=uncertain,
        inaccessible=inaccessible,
        sample_count=int(metadata[0]),
        voxel_size_mm=float(metadata[1]),
        reachable_distance_mm=float(metadata[2]),
        uncertain_distance_mm=float(metadata[3]),
        cutaway_quadrant=bool(metadata[7]),
        reachable_voxels=int(metadata[4]),
        uncertain_voxels=int(metadata[5]),
        inaccessible_voxels=int(metadata[6]),
    )
