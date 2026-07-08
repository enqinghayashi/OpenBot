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

