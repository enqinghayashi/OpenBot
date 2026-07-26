"""
Real-hardware grasp detection using DDS hand-state telemetry.

Replaces MuJoCo contact physics (grasp_detector.py) with motor-current
and tactile-sensor signals from the Dex3-1 hand via DDS subscriptions.

Usage:
    from real_grasp_detector import RealGraspDetector, GraspState

    detector = RealGraspDetector(object_mass=0.30)
    state = detector.update()
    if state.is_stable:
        print(f"Stable grasp: quadrant={state.gating_quadrant}")
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, MotorState_, PressSensorState_


# ---------------------------------------------------------------------------
# GraspState - the per-step grasp numbers
# ---------------------------------------------------------------------------

