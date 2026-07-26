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

@dataclass
class GraspState:
    """Per-step grasp quality metrics from DDS hand-state telemetry.

    Attributes
    ----------
    is_grasped : bool
        True when at least one tactile channel registers contact (>100000)
        OR distal motor torque exceeds threshold.
    contact_force_total : float
        Sum of abs(tau_est) across distal finger joints (2, 4, 6).
    contact_points : int
        Number of pressure sensor channels > 100000 across all pads.
    support_ratio : float
        Sum of distal joint tau_est magnitudes divided by object weight
        (mass * 9.81).  Values >= 1.0 mean motor torque alone could
        theoretically support the object against gravity.
    is_stable : bool
        True when the 4-quadrant gate is 'solid_grasp' or 'light_object'
        AND temporal stability counter >= stability_steps.
    motor_current_high : bool
        True when sum of distal tau_est > torque_threshold.
    tactile_contact : bool
        True when any pressure channel > pressure_threshold.
    gating_quadrant : str
        One of: 'solid_grasp', 'hit_obstacle', 'light_object', 'grasped_nothing'.
        Combined gate: motor_current_high X tactile_contact.
    motor_transient_detected : bool
        True when a negative spike (>0.03 N.m drop in one step) was detected
        in the sliding tau_est window during closure (replaces slip_velocity).
    contacts_detail : list[dict]
        Per-motor tau_est and per-pad pressure diagnostics.
    """
    is_grasped: bool = False
    contact_force_total: float = 0.0
    contact_points: int = 0
    support_ratio: float = 0.0
    is_stable: bool = False
    motor_current_high: bool = False
    tactile_contact: bool = False
    gating_quadrant: str = "grasped_nothing"
    motor_transient_detected: bool = False
    contacts_detail: List[Dict] = field(default_factory=list)
