"""pc_core: the pure-numpy geometry core of the point-cloud consumer.

stdlib + numpy ONLY (no rclpy, scipy, open3d, sklearn, grasp_geometry).
Python 3.8-safe.  Design: docs/POINTCLOUD_CONSUMER_DESIGN.md (batch 3, rev 2).

Five consumers, each returning a result dict
    {"status": OK|UNKNOWN|ERROR, "verdict": <member of VERDICTS>,
     "reason": <machine-readable, non-empty on UNKNOWN>, "evidence": {...}}
and wrapped by ``envelope()`` for the wire.

  P1 hold_baseline / hold_check : NEGATIVE-ONLY.  The consumer never asserts
     the object is held.  Decisive verdicts are STILL_ON_TABLE and MOVED, both
     negative.  ABSENT_UNPROVEN is NON-decisive ("no new evidence"); a caller
     must never read it as held.  P1 does NOT close run19 (bottle occluded ->
     no row -> no signal); that path is closed by the batch-2 CANCELLED fix.
  P2 fit_bowl      : rim height by a z-sweep circle fit (not z + h/2).
  P3 split_support : support split (support row, or a width LEDGE).
  P4 grasp_diag    : front surface / radius / corridor, DIAG only.
  P5 person_track  : robust lateral midpoint + range, SHADOW only.

Every geometric quantity is in the projector's torso frame:
  +x = RIGHT (projector :397 "'x': ... # lateral  (+x = right)"),
  +y = forward, +z = up.  The torso frame is NOT world-fixed; every task that
  compares or consumes geometry across time requires a MEASURED pose token.
"""
import json
import math

import numpy as np

try:  # package import
    from . import pc_config as _cfgmod
except ImportError:  # flat import
    # box layout / tests
    import pc_config as _cfgmod

PcConfig = _cfgmod.PcConfig
DEFAULT_CFG = _cfgmod.DEFAULT
cfg_hash = _cfgmod.cfg_hash

# -------------------------------------------------------------- the projector
# literals from
# skills/perception/depth_projector_node.py
# (rclpy/scipy, can't import)
# pinned by tests/unit/perception/
# test_pointcloud_synth.py (re-reads source)
FX, FY, CX, CY = 302.7, 302.8, 212.9, 121.1          # from projector :21
DEPTH_MIN = 0.3                                      # from projector :22
DEPTH_MAX = 6.0                                      # from projector :23
COLOR_W, COLOR_H = 424, 240                          # from projector :29
MAX_PTS_PER_OBJ = 2000                               # from projector :32
MAX_CLOUD_POINTS = 20000                             # from projector :37
TILT_DEG = 48.0                                      # from projector :54
_t = np.radians(TILT_DEG)
_R_base = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
_Rx_tilt = np.array([[1, 0, 0], [0, np.cos(_t), np.sin(_t)], [0, -np.sin(_t), np.cos(_t)]])
R_CAM_TO_TORSO = _R_base @ _Rx_tilt                  # from launch_pick_stack.sh:84
T_CAM_IN_TORSO = np.array([0.0, 0.04764, 0.46268])  # see yolo_node_seg.py:236

# sensor_msgs/PointField types
PF_UINT32 = 6
PF_FLOAT32 = 7
EXPECTED_FIELDS = (("x", 0, PF_FLOAT32, 1), ("y", 4, PF_FLOAT32, 1),
                   ("z", 8, PF_FLOAT32, 1), ("object_id", 12, PF_UINT32, 1))
CLOUD_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("oid", "<u4")])

# -------------------------------------------------------------- the verdicts
OK = "OK"
UNKNOWN = "UNKNOWN"
ERROR = "ERROR"
STATUSES = frozenset({OK, UNKNOWN, ERROR})

V_UNKNOWN = "UNKNOWN"
V_ERROR = "ERROR"
V_BASELINE_OK = "BASELINE_OK"
V_STILL = "STILL_ON_TABLE"
V_MOVED = "MOVED"
V_ABSENT = "ABSENT_UNPROVEN"
V_RIM = "RIM_FIT"
V_SPLIT = "SPLIT"
V_NOSPLIT = "NO_SPLIT_EVIDENCE"
V_GRASP = "GRASP_DIAG"
V_PERSON = "PERSON_TRACK"
VERDICTS = frozenset({V_UNKNOWN, V_ERROR, V_BASELINE_OK, V_STILL, V_MOVED, V_ABSENT,
                      V_RIM, V_SPLIT, V_NOSPLIT, V_GRASP, V_PERSON})
# P1: negative-only
# no HOLD verdict claims held
HOLD_VERDICTS = frozenset({V_STILL, V_MOVED, V_ABSENT, V_UNKNOWN})
HOLD_DECISIVE_NEGATIVE = frozenset({V_STILL, V_MOVED})
HOLD_NON_DECISIVE = frozenset({V_ABSENT, V_UNKNOWN})

PHASES = ("tucked", "cancelled", "error", "other")
TASKS = ("hold_baseline", "hold_check", "place_rim", "split", "grasp", "person")

POSE_KEYS = ("torso_z", "pitch", "roll", "waist_pitch")

# keep states apart
NO_PACKET = "NO_PACKET"
NO_ROW = "NO_ROW"
NO_POINTS = "NO_POINTS"
TRUNCATED = "TRUNCATED"
SEL_OK = "OK"


class CloudFormatError(ValueError):
    pass


