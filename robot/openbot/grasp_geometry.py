"""
Contact-Aware Palm x_off for Humanoid Power Grasping
=====================================================

Given the G1's depth camera + object detection pipeline, compute the optimal
wrist offset (x_off) for a rich-contact Dex3 power grasp.

Ported from palm_x_off.py with ZERO MuJoCo dependency — pure numpy.

Mathematical Pipeline
---------------------

::

    RGB + Depth + detection mask
           │
           ▼
    Object point cloud (3-D, world/torso frame)
           │
           ▼
    Geometric primitive fitting  (cylinder / sphere / box / PCA-ellipsoid)
           │
           ▼
    Cross-section at grasp_z along approach lane
           │
           ▼
    For each candidate x_off ∈ [-0.20, -0.02]:
        │
        ├─ Compute Dex3 contact positions (palm, index, middle, thumb)
        ├─ Compute penetration depth into object surface
        ├─ Estimate contact area
        ├─ Score: palm_contact × finger_wrap × width_compatibility
        │
        ▼
    Select x_off* = argmax score(x_off)
           │
           ▼
    Output: (x_off*, grasp_point, quality_score, contact_area)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# ── Dex3 Hand Geometry (metres) ──
HAND_LENGTH_PALM   = 0.042   # wrist_yaw_link → palm middle
HAND_LENGTH_FINGER = 0.119   # palm centre → middle fingertip
HAND_LENGTH_TOTAL  = 0.161   # wrist → fingertip
PALM_WIDTH         = 0.080   # palm pad width
THUMB_LENGTH       = 0.070   # thumb reach
MAX_APERTURE       = 0.077   # thumb↔index at HAND_OPEN
GRASP_Z_DEFAULT    = 0.90    # wrist Z at grasp (above table)

# ── Calibration ──
CALIBRATION_RADIUS  = 0.035   # 7 cm bottle, our anchor
CALIBRATION_X_OFF   = -0.100  # calibrated x_off for the bottle
S_INSERTION         = -CALIBRATION_X_OFF - HAND_LENGTH_PALM - CALIBRATION_RADIUS
# 0.100 - 0.042 - 0.035 = 0.023 m standoff

# ── Scoring Weights ──
W_PALM  = 0.40   # palm contact weight
W_WRAP  = 0.35   # finger wrap weight
W_WIDTH = 0.25   # width fit weight

# ── Search Parameters ──
X_OFF_MIN  = -0.20   # fully retracted
X_OFF_MAX  = -0.02   # fully forward
X_OFF_STEP =  0.005  # 5 mm per step

# ── Object Radius Lookup (used when there's no point cloud) ──
OBJECT_RADII: Dict[str, float] = {
    "bottle":       0.035,  # it's a 7 cm bottle
    "cup":          0.040,  # 8 cm across
    "mouse":        0.030,  # 6 cm wide
    "cell phone":   0.035,  # 7 cm wide
    "book":         0.010,  # 2 cm thick
    "apple":        0.040,  # 8 cm across
    "square_block": 0.050,  # 10 cm wide
    "camera":       0.030,  # 6 cm wide
    "container":    0.125,  # 25 cm long
}

OBJECT_SHAPES: Dict[str, str] = {
    "bottle":       "cylinder",
    "cup":          "cylinder",
    "mouse":        "box",
    "cell phone":   "box",
    "book":         "box",
    "apple":        "sphere",
    "square_block": "box",
    "camera":       "box",
    "container":    "cylinder",
}

# ── Robot ──
ROBOT_ORIGIN_XY = np.array([0.0, 0.0], dtype=np.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraspQuality:
    """Output of the contact-scoring optimisation."""
    x_off: float               # wrist offset (m)
    quality: float             # overall score, [0, 1]
    q_palm: float              # sub-score: palm contact
    q_wrap: float              # sub-score: finger wrap
    q_width: float             # sub-score: width fit
    contact_area: float        # contact area (m²)
    palm_penetration: float    # palm sink depth (m)
    finger_reach_past: float   # past far surface (m); <0 = short
    grasp_point: np.ndarray    # (3,) wrist target, world
    lane_dir: np.ndarray       # (3,) lane direction
    primitive_type: str        # fitted shape: cylinder|sphere|box|pca

    @property
    def is_good(self) -> bool:
        """Heuristic: quality ≥ 0.5 is a viable grasp."""
        return self.quality >= 0.50

    @property
    def is_excellent(self) -> bool:
        """Heuristic: quality ≥ 0.75 is a confident grasp."""
        return self.quality >= 0.75


@dataclass
class GeometricPrimitive:
    """Fitted geometric model of the object from point cloud data."""
    type: str                        # cylinder|sphere|box|pca
    centre: np.ndarray               # (3,) in world frame
    # cylinder stuff
    axis: Optional[np.ndarray] = None     # (3,) unit axis
    radius: Optional[float] = None        # in metres
    height: Optional[float] = None        # in metres too
    # sphere stuff
    sphere_radius: Optional[float] = None
    # box stuff
    half_extents: Optional[np.ndarray] = None  # (3,) half-widths per side
    rotation: Optional[np.ndarray] = None      # (3,3) box rotation
    # PCA params
    pca_eigenvalues: Optional[np.ndarray] = None  # (3,) variance per axis
    pca_axes: Optional[np.ndarray] = None          # (3,3) principal axes


# ═══════════════════════════════════════════════════════════════════════════════
# Geometric Primitive Fitting
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_shape(eigenvalues: np.ndarray) -> str:
    """Classify shape from eigenvalue ratios.

    - Cylinder: one eigenvalue >> others  (elongated along one axis)
    - Sphere: all eigenvalues similar
    - Box: two eigenvalues similar, third intermediate
    """
    e = np.sqrt(np.maximum(eigenvalues, 1e-12))  # std devs turn into half-extents
    e_sorted = np.sort(e)[::-1]

    if e_sorted[2] < 1e-6:
        return "pca"   # degenerate, nothing to classify

    ratio_1_2 = e_sorted[0] / e_sorted[1]
    ratio_2_3 = e_sorted[1] / e_sorted[2]

    if ratio_1_2 > 2.5:
        return "cylinder"   # clearly elongated
    elif ratio_2_3 > 2.5:
        return "box"        # flat on one axis
    elif max(ratio_1_2, ratio_2_3) < 1.8:
        return "sphere"
    else:
        return "pca"


def _fit_cylinder(
    points: np.ndarray, centre: np.ndarray,
    eigenvectors: np.ndarray, eigenvalues: np.ndarray,
) -> GeometricPrimitive:
    """Fit cylinder: primary PCA axis = cylinder axis, radius from orthogonal distances."""
    axis = eigenvectors[:, 0].copy()  # grab the primary axis
    axis /= np.linalg.norm(axis)

    # distance from the axis
    centred = points - centre
    axial_proj = (centred @ axis)
    perpendicular = centred - np.outer(axial_proj, axis)
    radial_distances = np.linalg.norm(perpendicular, axis=1)

    # 85th percentile: outliers shouldn't skew it
    radius = float(np.percentile(radial_distances, 85))
    radius = max(radius, 0.005)  # never smaller than 5 mm

    # axial stretch
    height = float(np.max(axial_proj) - np.min(axial_proj))
    height = max(height, 2 * radius)  # some sane minimum

    return GeometricPrimitive(
        type="cylinder", centre=centre,
        axis=axis, radius=radius, height=height,
    )


def _fit_sphere(points: np.ndarray, centre: np.ndarray) -> GeometricPrimitive:
    """Fit sphere: centre = mean, radius = mean distance from centre."""
    distances = np.linalg.norm(points - centre, axis=1)
    radius = float(np.mean(distances))
    radius = max(radius, 0.005)
    return GeometricPrimitive(
        type="sphere", centre=centre, sphere_radius=radius,
    )


def _fit_box(points: np.ndarray, centre: np.ndarray,
             eigenvectors: np.ndarray) -> GeometricPrimitive:
    """Fit oriented box: project onto PCA axes, half-extents = 90th percentile spans."""
    centred = points - centre
    projections = centred @ eigenvectors  # (N, 3), now in the PCA frame

    half_extents = np.zeros(3, dtype=np.float64)
    for i in range(3):
        lo = float(np.percentile(projections[:, i], 5))
        hi = float(np.percentile(projections[:, i], 95))
        half_extents[i] = max((hi - lo) / 2.0, 0.005)

    return GeometricPrimitive(
        type="box", centre=centre,
        half_extents=half_extents, rotation=eigenvectors.copy(),
        # fallback radius: smaller minor half-extent
        radius=float(np.min(half_extents[1:])),
    )


def fit_geometric_primitive(
    points: np.ndarray,
    hint_shape: Optional[str] = None,
) -> GeometricPrimitive:
    """Fit a geometric primitive to a 3-D point cloud.

    Parameters
    ----------
    points : (N, 3) float64
        Object point cloud in world frame.
    hint_shape : str or None
        Expected shape: ``"cylinder"``, ``"sphere"``, ``"box"``, or None for auto.

    Returns
    -------
    GeometricPrimitive
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 4:
        # too few points: sphere from the extent
        centre = points.mean(axis=0) if points.shape[0] > 0 else np.zeros(3)
        r = float(np.max(np.linalg.norm(points - centre, axis=1)))
        return GeometricPrimitive(type="sphere", centre=centre,
                                  sphere_radius=max(r, 0.01))

    centre = points.mean(axis=0)
    cov = np.cov((points - centre).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # descending: primary axis = largest eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # now decide the shape
    if hint_shape is None:
        hint_shape = _classify_shape(eigenvalues)

    if hint_shape == "cylinder":
        return _fit_cylinder(points, centre, eigenvectors, eigenvalues)
    elif hint_shape == "sphere":
        return _fit_sphere(points, centre)
    elif hint_shape == "box":
        return _fit_box(points, centre, eigenvectors)
    else:
        # PCA ellipsoid, good enough
        r = np.sqrt(eigenvalues)  # std devs
        return GeometricPrimitive(
            type="pca", centre=centre,
            pca_eigenvalues=eigenvalues, pca_axes=eigenvectors,
            radius=float(np.mean(r[:2])),  # radius from the minor axes
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Section Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_local_surface_radius(
    primitive: GeometricPrimitive,
    lane_dir: np.ndarray,
    grasp_z: float = GRASP_Z_DEFAULT,
) -> float:
    """Compute the effective surface curvature radius at the approach point.

    This is the radius r_c such that the object surface at the approach
    intersection behaves locally like a circle of radius r_c.

    Parameters
    ----------
    primitive : GeometricPrimitive
    lane_dir : (3,) unit vector, approach direction (robot → object)
    grasp_z : float, wrist Z height

    Returns
    -------
    r_c : float
        Local curvature radius along the approach lane (metres).
    """
    centre_z = primitive.centre[2]

    if primitive.type == "cylinder" and primitive.axis is not None:
        axis = primitive.axis / np.linalg.norm(primitive.axis)
        axis_verticality = abs(axis[2])

        if axis_verticality > 0.95:
            # vertical cylinder: slice is a circle
            return primitive.radius

        # tilted: ellipse, a = r / sin(φ)
        radius_at_z = primitive.radius / max(axis_verticality, 0.1)
        return radius_at_z

    elif primitive.type == "sphere" and primitive.sphere_radius is not None:
        R = primitive.sphere_radius
        dz = grasp_z - centre_z
        if abs(dz) >= R:
            return max(R * 0.3, 0.01)
        r_at_z = np.sqrt(max(R * R - dz * dz, 0.0))
        return r_at_z

    elif primitive.type == "box" and primitive.half_extents is not None:
        if primitive.rotation is not None:
            lane_local = primitive.rotation.T @ lane_dir  # lane, in the box frame
            h = primitive.half_extents
            box_depth = np.sum(np.abs(lane_local) * h)
        else:
            box_depth = primitive.half_extents[0]  # just assume X is the depth
        return box_depth

    elif primitive.type == "pca" and primitive.radius is not None:
        return primitive.radius

    # no match: fall back
    return 0.035  # bottle radius default


def compute_cross_section_width(
    primitive: GeometricPrimitive,
    lane_dir: np.ndarray,
    grasp_z: float = GRASP_Z_DEFAULT,
) -> float:
    """Compute the object's width PERPENDICULAR to the approach lane.

    This determines whether the Dex3 hand can span the object for a power
    grasp (thumb on one side, fingers on the other).

    Returns width in metres.
    """
    # perp in the XY plane
    perp = np.array([-lane_dir[1], lane_dir[0], 0.0], dtype=np.float64)

    if primitive.type == "cylinder" and primitive.axis is not None:
        axis = primitive.axis / np.linalg.norm(primitive.axis)
        if abs(axis[2]) > 0.95:
            return 2.0 * primitive.radius
        return 2.0 * primitive.radius

    elif primitive.type == "sphere" and primitive.sphere_radius is not None:
        R = primitive.sphere_radius
        dz = grasp_z - primitive.centre[2]
        if abs(dz) >= R:
            return R * 0.5
        r_at_z = np.sqrt(max(R * R - dz * dz, 0.0))
        return 2.0 * r_at_z

    elif primitive.type == "box" and primitive.half_extents is not None:
        if primitive.rotation is not None:
            perp_local = primitive.rotation.T @ perp
            width = 2.0 * np.sum(np.abs(perp_local) * primitive.half_extents)
        else:
            width = 2.0 * max(primitive.half_extents[1], primitive.half_extents[2])
        return width

    elif primitive.type == "pca" and primitive.radius is not None:
        return 2.0 * primitive.radius

    return 0.07


# ═══════════════════════════════════════════════════════════════════════════════
# Contact Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_candidate_x_off(
    x_off: float,
    r_c: float,           # local curvature radius (m)
    object_width: float,  # width across the lane (m)
    object_centre_z: float,
    grasp_z: float = GRASP_Z_DEFAULT,
) -> Tuple[float, float, float, float, float, float]:
    """Score a candidate x_off for grasp quality, modelling the insertion process.

    **Contact point (post-insertion).**  After the insertion converges, the
    palm contacts the near surface::

        x_off_contact = -(r_c + HAND_LENGTH_PALM)

    **Anchor point.**  The optimal x_off is BEHIND x_off_contact by the
    insertion standoff S_INSERTION::

        x_off_anchor = x_off_contact - S_INSERTION
                     = -(r_c + HAND_LENGTH_PALM + S_INSERTION)

    For the 7 cm bottle (r_c=0.035)::

        x_off_anchor = -(0.035 + 0.042 + 0.023) = -0.100

    Parameters
    ----------
    x_off : float
        Candidate wrist offset along lane from object centre (metres).
    r_c : float
        Local surface curvature radius at the approach point (metres).
    object_width : float
        Object width perpendicular to lane (metres).
    object_centre_z : float
        Object centre Z in world frame.
    grasp_z : float
        Wrist Z height at grasp.

    Returns
    -------
    (quality, q_palm, q_wrap, q_width, contact_area, palm_penetration)
    """
    # ── contact geometry (post-insertion) ──
    x_off_contact = -(r_c + HAND_LENGTH_PALM)        # wrist at the near surface
    x_off_anchor = x_off_contact - S_INSERTION        # where the insertion starts
    near_surface_lane = -r_c
    far_surface_lane = r_c

    # ── Q_palm: how well this x_off supports the insertion approach ──
    insertion_room = x_off_contact - x_off  # +ve = behind contact

    MAX_INSERTION_ROOM = 0.10  # 10 cm behind - too much, out of reach
    MIN_INSERTION_ROOM = 0.005 # 5 mm - barely any room at all

    if insertion_room <= 0:
        q_palm = 0.0
    elif insertion_room <= MIN_INSERTION_ROOM:
        q_palm = insertion_room / MIN_INSERTION_ROOM  # ramps up 0 → 1
    elif insertion_room <= S_INSERTION:
        q_palm = 1.0 - 0.1 * (S_INSERTION - insertion_room) / (S_INSERTION - MIN_INSERTION_ROOM + 1e-9)
        q_palm = max(0.7, q_palm)
    elif insertion_room <= MAX_INSERTION_ROOM:
        q_palm = 1.0 - 0.8 * (insertion_room - S_INSERTION) / (MAX_INSERTION_ROOM - S_INSERTION)
        q_palm = max(0.2, q_palm)
    else:
        q_palm = 0.2

    # ── Q_wrap: how well the fingers wrap around after insertion ──
    finger_reach_past_post = HAND_LENGTH_FINGER - 2.0 * r_c

    MIN_WRAP = 0.005   # 5 mm past the far surface
    IDEAL_WRAP = 0.025  # 2.5 cm past the far surface

    if finger_reach_past_post <= 0:
        q_wrap = 0.0
    elif finger_reach_past_post <= MIN_WRAP:
        q_wrap = finger_reach_past_post / MIN_WRAP  # ramps up 0 → 1
    elif finger_reach_past_post <= IDEAL_WRAP:
        q_wrap = 1.0
    else:
        q_wrap = 1.0  # great wrap, full marks

    # ── Q_width: can the hand actually span the object? ──
    aperture_available = MAX_APERTURE * 1.05  # a little tolerance
    if object_width <= aperture_available:
        IDEAL_WIDTH = 0.050  # 5 cm ideal for a wrap
        deviation = abs(object_width - IDEAL_WIDTH) / IDEAL_WIDTH
        q_width = max(0.0, 1.0 - 0.5 * deviation)
    else:
        excess = (object_width - aperture_available) / aperture_available
        q_width = np.exp(-4.0 * excess)  # exp penalty when too wide

    # ── rough contact area (post-insertion) ──
    effective_palm_pen = 0.008  # 8 mm typical palm sink
    palm_contact_area = PALM_WIDTH * effective_palm_pen * min(q_palm, 1.0)

    n_fingers = 2 if q_wrap > 0.5 else (1 if q_wrap > 0.2 else 0)
    finger_contact_area = n_fingers * 0.012 * 0.008  # call it ~1cm² per finger

    contact_area = palm_contact_area + finger_contact_area

    # penetration before insertion; diagnostics only
    palm_lane_pre = x_off + HAND_LENGTH_PALM
    palm_penetration_pre = palm_lane_pre - near_surface_lane

    # ── and the overall quality ──
    quality = W_PALM * q_palm + W_WRAP * q_wrap + W_WIDTH * q_width

    return float(quality), float(q_palm), float(q_wrap), float(q_width), \
           float(contact_area), float(palm_penetration_pre)
