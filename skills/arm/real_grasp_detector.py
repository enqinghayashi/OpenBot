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

    def reset(self):
        """Reset internal counters (call between pick attempts)."""
        self._stable_counter = 0
        self._tau_window.clear()
        self._prev_tau_sum = None

    def check_palm_contact(self, pressure_threshold: float = None) -> bool:
        """Check if the palm is contacting an object (hand open, wrist advancing).

        During insertion the hand is open — fingers are extended and cannot
        trigger fingertip pressure sensors.  Any pressure reading above
        threshold therefore indicates PALM contact.

        The Dex3-1 has 3 palm packages × 2×2 array = 12 palm sensors
        (documented in the Dex3-1 user manual).  This method checks ALL
        pressure packages — during open-hand insertion, only the palm
        sensors can register contact.

        Debug output on first call prints sensor layout; every ~50th call
        prints max pressure so you can verify the DDS subscription is live.

        Returns True if any pressure channel exceeds the threshold.
        """
        threshold = pressure_threshold if pressure_threshold is not None else self._pressure_threshold
        hand_state = self._read_state()
        if hand_state is None:
            return False

        max_pressure = 0.0
        trigger_pkg = -1
        trigger_ch = -1
        n_pkgs = 0
        for pkg in hand_state.press_sensor_state:
            n_pkgs += 1
            for ch, p in enumerate(pkg.pressure):
                if p > max_pressure:
                    max_pressure = p
                if p > threshold:
                    trigger_pkg = pkg_idx = n_pkgs - 1
                    trigger_ch = ch
                    # no break: max over all pkgs

        # debug: layout, then max now and then
        if not hasattr(self, '_palm_debug_count'):
            self._palm_debug_count = 0
            import sys as _sys
            _sys.stderr.write(
                f'[RealGraspDetector {self._side}] palm sensors: '
                f'{n_pkgs} pressure packages detected, '
                f'threshold={threshold:.0f}\n')
            _sys.stderr.flush()
        self._palm_debug_count += 1
        if self._palm_debug_count % 50 == 0:
            import sys as _sys
            _sys.stderr.write(
                f'[RealGraspDetector {self._side}] palm debug #{self._palm_debug_count}: '
                f'max_pressure={max_pressure:.0f} (threshold={threshold:.0f}), '
                f'{n_pkgs} pkgs\n')
            _sys.stderr.flush()

        if trigger_pkg >= 0:
            import sys as _sys
            _sys.stderr.write(
                f'[RealGraspDetector {self._side}] PALM CONTACT DETECTED: '
                f'pkg={trigger_pkg} ch={trigger_ch} '
                f'pressure={max_pressure:.0f} > threshold={threshold:.0f}\n')
            _sys.stderr.flush()
            return True
        return False

    def check_middle_finger_contact(self, threshold: float = None) -> bool:
        """Check if the middle finger has pressure contact.

        Checks ONLY packages 5 (middle finger pad) and 8 (middle fingertip).
        Returns True if any channel in those two packages exceeds the
        threshold.

        Debug: on the first call, prints the package indices being checked
        and the threshold.  On every detection, prints which package and
        channel triggered.

        Parameters
        ----------
        threshold : float, optional
            Pressure threshold.  Defaults to ``self._pressure_threshold``
            (110000 — tuned to fire above DDS cross-talk at ~103k).

        Returns
        -------
        bool
            True if any middle-finger pressure channel exceeds threshold.
        """
        threshold = threshold if threshold is not None else self._pressure_threshold
        hand_state = self._read_state()
        if hand_state is None:
            return False

        pkgs = hand_state.press_sensor_state
        middle_pkg_indices = [5, 8]  # 5 = pad, 8 = tip

        # debug: packages, once
        if not hasattr(self, '_middle_debug_count'):
            self._middle_debug_count = 0
            import sys as _sys
            _sys.stderr.write(
                f'[RealGraspDetector {self._side}] middle finger contact: '
                f'checking packages {middle_pkg_indices}, threshold={threshold:.0f}\n')
            _sys.stderr.flush()

        self._middle_debug_count += 1

        for pkg_idx in middle_pkg_indices:
            if pkg_idx >= len(pkgs):
                continue
            for ch, p in enumerate(pkgs[pkg_idx].pressure):
                if p > threshold:
                    import sys as _sys
                    _sys.stderr.write(
                        f'[RealGraspDetector {self._side}] MIDDLE FINGER CONTACT: '
                        f'pkg={pkg_idx} ch={ch} pressure={p:.0f} > threshold={threshold:.0f}\n')
                    _sys.stderr.flush()
                    return True

        return False

    def check_palm_and_middle(
        self, palm_threshold: float = None, middle_threshold: float = None
    ):
        """Check palm contact and middle-finger contact in one call.

        Palm contact is checked via :meth:`check_palm_contact` (scans ALL
        pressure packages — during open-hand insertion only the palm can
        register contact).  Middle-finger contact uses
        :meth:`check_middle_finger_contact` (scans only packages 5 + 8).

        Parameters
        ----------
        palm_threshold : float, optional
            Threshold forwarded to ``check_palm_contact``.
        middle_threshold : float, optional
            Threshold forwarded to ``check_middle_finger_contact``.

        Returns
        -------
        tuple[bool, bool]
            ``(palm_contact, middle_contact)``.
        """
        palm = self.check_palm_contact(pressure_threshold=palm_threshold)
        middle = self.check_middle_finger_contact(threshold=middle_threshold)
        return (palm, middle)

    def get_max_pressure(self) -> float:
        """Return the maximum pressure value across ALL sensor packages.

        Returns 0 if no DDS data is available.  Used by adaptive closure
        to track pressure increase as fingers close around the object.
        """
        hand_state = self._read_state()
        if hand_state is None:
            return 0.0
        max_p = 0.0
        for pkg in hand_state.press_sensor_state:
            for p in pkg.pressure:
                if p > max_p:
                    max_p = float(p)
        return max_p

    def get_finger_pressure(self) -> float:
        """Return the maximum pressure across finger sensor packages only.

        Excludes palm packages (indices 0-2).  Returns 0 if no DDS data.
        Used by adaptive closure to track finger grip pressure independently
        from palm contact pressure.
        """
        hand_state = self._read_state()
        if hand_state is None:
            return 0.0
        max_p = 0.0
        pkgs = hand_state.press_sensor_state
        n_finger_pkgs = 0
        # 3-5 pads, 6-8 tips
        for pkg_idx in range(3, min(len(pkgs), 9)):
            n_finger_pkgs += 1
            for p in pkgs[pkg_idx].pressure:
                if p > max_p:
                    max_p = float(p)

        # debug: per-pkg max, every 25 calls
        if not hasattr(self, '_fp_debug_count'):
            self._fp_debug_count = 0
        self._fp_debug_count += 1
        if self._fp_debug_count % 25 == 0:
            pkg_maxes = []
            for pkg_idx in range(3, min(len(pkgs), 9)):
                pkg_max = max(pkgs[pkg_idx].pressure) if pkgs[pkg_idx].pressure else 0.0
                pkg_maxes.append(f'pkg[{pkg_idx}]={pkg_max:.0f}')
            import sys as _sys
            _sys.stderr.write(
                f'[RealGraspDetector {self._side}] finger_pressure=#{self._fp_debug_count}: '
                f'n_finger_pkgs={n_finger_pkgs}, max={max_p:.0f}, '
                f'per_pkg=[{", ".join(pkg_maxes)}]\n')
            _sys.stderr.flush()

        return max_p

    def diagnose_palm_sensors(self):
        """One-shot diagnostic: print all pressure sensor packages and their max values.

        Call this once after daemon init to verify the DDS subscription is live
        and the dex3_server is publishing hand state data.
        """
        import sys as _sys
        hand_state = self._read_state()
        if hand_state is None:
            _sys.stderr.write(
                f'[RealGraspDetector {self._side}] DIAGNOSE: '
                f'NO hand state received — is dex3_server running? '
                f'Topic: rt/dex3/{self._side}/state\n')
            _sys.stderr.flush()
            return False

        n_pkgs = len(hand_state.press_sensor_state)
        n_motors = len(hand_state.motor_state)
        _sys.stderr.write(
            f'[RealGraspDetector {self._side}] DIAGNOSE: '
            f'{n_pkgs} pressure packages, {n_motors} motors\n')
        for pkg_idx, pkg in enumerate(hand_state.press_sensor_state):
            max_p = max(pkg.pressure) if pkg.pressure else 0.0
            active_ch = sum(1 for p in pkg.pressure if p > self._pressure_threshold)
            _sys.stderr.write(
                f'[RealGraspDetector {self._side}]   pkg[{pkg_idx}]: '
                f'max={max_p:.0f}, active_channels={active_ch}/12 '
                f'(threshold={self._pressure_threshold:.0f})\n')
        _sys.stderr.flush()
        return True

