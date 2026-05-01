"""Persistent scene graph: merges event-driven detections into stable object refs that survive walks."""

import math
import re
import time
from typing import Dict, List, Optional

from brain.observation import Detection, WorldObservation
from brain.verifier import Verdict
from brain.self_state import SelfState, PLACE_RELEASED_MARK
from brain import spatial

# ref dict, shared contract.
# extra optional key: instance_id: int
# (see _claim_instance).
#   label: str          normalized, lowercase
#   x, y, z: float      last torso pose (m)
#   conf: float         newest detection confidence
#   last_seen: float    ts of newest confirmation (s)
#   holding: bool       believed in gripper
#   in_container: bool|None  post-place belief
#   color_name: str|None  perception colour, prose only
#   body_side: str|None  left/right, prose only
#   colour_vote: list   last 7 colour names
#   display_name: str|None  spatial.display_names

# '<label>_<n>' is an EXACT-REF ask.
# No substring fallthrough: a missing
# 'bottle_10' must never alias 'bottle_1'.
# Same inside a phrase: 'grab bottle_10'.
_REF_SYNTAX_RE = re.compile(r"^[a-z0-9_]+_\d+$")

#: Planar standoff (m). Past it the engine refuses.
#:
#: Residual by target y: 0.057@0.249,
#: 0.107@0.435, 0.352@0.629,
#: 0.348@0.750, 0.397@0.771 m.
#: Refusal bar 0.15 m (REACH_REFUSE_POS,
#: skills/arm/pick_engine_v3.py).
#: Slope 1.26 m/m -> 0.15 at y ~= 0.47 m.
#: 0.50 leaves ~3 cm.
#:
#: Caveat: needs the full 3D pose and stance,
#: not planar y alone. A hint, not a promise.
PICK_REACH_ADVISE_M = 0.50

#: OUT-OF-REACH uses the engine's reach LAW, not the hypot:
#: residual = d(right shoulder -> target) - 0.382 m.
#: Shoulder in torso_link: (fwd 0.00396, right 0.10021,
#: up 0.24778) = (0.10021, 0.00396, 0.24778) here.
#: Refusal above 0.15 m (pick_engine_v3.py).
#: Hypot missed 0.10 m right offset, 0.25 m height:
#: cup (0.284, 0.428, -0.010), hypot 0.514,
#: residual 0.147 -- reachable, model walked away.
#: Left Dex3 is out; right arm is the arm.
RIGHT_SHOULDER_XYZ = (0.10021, 0.00396, 0.24778)
ARM_REACH_LAW_M = 0.382
REACH_REFUSE_POS_M = 0.15


def reach_residual(x: float, y: float, z: float) -> float:
    """Predicted pick residual (m) of the right arm for a torso-frame target."""
    sx, sy, sz = RIGHT_SHOULDER_XYZ
    return math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) - ARM_REACH_LAW_M
