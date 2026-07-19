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


# ---------------------------------------------------------------------------
# RealGraspDetector - the detector itself
# ---------------------------------------------------------------------------

class RealGraspDetector:
    """Detect and monitor grasp quality from live DDS hand-state telemetry.

    Subscribes to ``rt/dex3/{side}/state`` and reads ``MotorState_``
    (tau_est for torque sensing) and ``PressSensorState_`` (pressure
    arrays for tactile sensing).

    Parameters
    ----------
    side : str
        Which Dex3-1 hand to monitor: ``'right'`` or ``'left'``
        (default ``'right'`` for backward compatibility).
    object_mass : float
        Mass in kg for support_ratio normalisation (default 0.30).
    torque_threshold : float
        Sum of distal tau_est magnitudes above which motor_current_high
        is True (default 0.05 N.m).
    pressure_threshold : float
        Pressure value above which a channel is considered in contact
        (default 100000).
    stability_steps : int
        Consecutive updates with gating in {'solid_grasp','light_object'}
        before is_stable=True (default 5).
    dt : float
        Expected time step for transient detector window (default 0.02 s).
    """

    # distal joints in the Dex3-1 7-motor sequence: thumb distal is
    # motor 2 on BOTH hands; 4 and 6 are the index/middle distals, but
    # which is which depends on the hand.  Right: 4=index_1, 6=middle_1;
    # left: 4=middle_1, 6=index_1 — see grip_server.DEX3_JOINT_NAMES.
    # We only ever SUM this tuple, so it doesn't matter.
    DISTAL_JOINT_IDS = [2, 4, 6]

    def __init__(
        self,
        side: str = "right",
        object_mass: float = 0.30,
        torque_threshold: float = 0.05,
        pressure_threshold: float = 130000,
        tactile_verify_threshold: float = 100000,
        finger_pad_threshold: float = 100000,
        fingertip_threshold: float = 100000,
        stability_steps: int = 5,
        dt: float = 0.02,
    ):
        self._side = side
        self._object_mass = object_mass
        self._object_weight = object_mass * 9.81  # newtons
        self._torque_threshold = torque_threshold
        self._pressure_threshold = pressure_threshold             # palm (pkgs 0-2)
        self._tactile_verify_threshold = tactile_verify_threshold # grasp verify
        self._finger_pad_threshold = finger_pad_threshold         # pads (pkgs 3-5)
        self._fingertip_threshold = fingertip_threshold           # tips (pkgs 6-8)
        self._stability_steps = stability_steps
        self._dt = dt

        # rt/dex3/{side}/state
        self._sub = ChannelSubscriber(f"rt/dex3/{side}/state", HandState_)
        self._sub.Init(handler=None, queueLen=10)

        # good ticks
        self._stable_counter: int = 0

        # sliding tau window, ~0.2 s at dt=0.02
        # -> 10 samples
        self._tau_window: deque = deque(maxlen=max(10, int(0.2 / dt)))
        self._prev_tau_sum: Optional[float] = None

    # ------------------------------------------------------------------
    # the public API (what the protocol loop calls)
    # ------------------------------------------------------------------

    def update(self) -> GraspState:
        """Read latest DDS HandState_ and return grasp metrics.

        Call at ~50 Hz from the main protocol loop.  Non-blocking:
        returns immediately with the latest state, or a default
        GraspState if no message is available yet.
        """
        hand_state = self._read_state()
        if hand_state is None:
            return GraspState()

        motor_states = hand_state.motor_state
        press_sensor_states = hand_state.press_sensor_state

        # 7 motors or bail
        if len(motor_states) < 7:
            return GraspState()

        # 1. motor current
        tau_sum, motor_high = self._compute_torque_support(motor_states)

        # 2. tactile
        contact_count, tactile_contact_flag = self._compute_tactile_contact(
            press_sensor_states)

        # 3. slip spike?
        transient = self._detect_transient(tau_sum)

        # 4. rest
        is_grasped = tactile_contact_flag or motor_high
        support_ratio = (
            tau_sum / self._object_weight if self._object_weight > 0 else 0.0
        )

        # 5. 2x2 gate
        quadrant = self._combined_gating(motor_high, tactile_contact_flag)

        # 6. good run
        if quadrant in ("solid_grasp", "light_object"):
            self._stable_counter += 1
        else:
            self._stable_counter = 0
        is_stable = self._stable_counter >= self._stability_steps

        # per-motor detail
        details = self._build_details(motor_states, press_sensor_states)

        return GraspState(
            is_grasped=is_grasped,
            contact_force_total=tau_sum,
            contact_points=contact_count,
            support_ratio=support_ratio,
            is_stable=is_stable,
            motor_current_high=motor_high,
            tactile_contact=tactile_contact_flag,
            gating_quadrant=quadrant,
            motor_transient_detected=transient,
            contacts_detail=details,
        )

