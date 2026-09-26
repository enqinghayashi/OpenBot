"""arm_guard.py — upper-body owner: poses, cosine slew, gate 29, trailing hold.

Poses are the hardware-verified values from helpers/manual_poses.json —
EXCEPT "HOME", which is sim-verified only (see its comment) —
(7-DOF per arm, motor order 15-21 left / 22-28 right: shoulder_pitch,
shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw).

Slew = cosine interpolation at arm_slew_rate rad/s, min arm_slew_min_s —
the daemon's move_arm_smooth convention. Deliberately NOT the 100-150 ms
coupling ramp: full-pose slews must take >= 1.5 s (0.3 rad/s worst joint).
"""

import math

import numpy as np

L_OFF, R_OFF = 15, 22          # our slots in the 35-slot command
ARM_SPAN = 14                  # 7 left + 7 right

# from helpers/manual_poses.json, checked on hardware
ARM_POSES = {
    "READY": {
        "left": [0.40, 0.30, 0.50, -1.00, 0.00, 0.00, -0.01],
        "right": [0.40, -0.30, -0.50, -1.00, 0.00, 0.00, 0.01],
    },
    "LIFT": {
        "left": [0.00, 0.20, 0.30, -1.00, 0.00, -0.60, 0.20],
        "right": [0.00, -0.20, -0.30, -1.00, 0.00, -0.60, -0.20],
    },
    "CARRY_TUCK": {
        "left": [0.50, 0.05, 0.00, 0.00, -1.50, 0.00, -1.00],
        "right": [0.50, -0.05, 0.00, 0.00, 1.50, 0.00, 1.00],
    },
    "NEUTRAL": {
        # canonical neutral, symmetric arms, same roll
        # as CARRY_TUCK; writer slews here at boot, so keep byte-identical
        # to pick_engine_v2.py
        "left": [0.25522, 0.23, -0.08654, 0.83254, -1.50, 0.01032, -0.00063],
        "right": [0.25522, -0.23, -0.08654, 0.83254, 1.50, 0.01032, -0.00063],
    },
    # HOME — operator ask: both arms hang, palms on the thighs,
    # thumbs forward.
    #
    # !! NOT hardware-verified unlike the poses above — only SIM-VERIFIED, on
    # skills/writer/sim/g1_gear_wbc.xml (md5 8ffca967c299400100d80d2ce40a917f),
    # root z=0.793. Go measure it on Stuart (manual_arm_joints.py --pose home)
    # before trusting it.
    #
    # measured, not on paper:
    #   shoulder_pitch 0.00 — upper arm already hangs 5.6 deg forward of
    #     plumb at q=0 (URDF shoulder offset, dir (0.085, 0.034, -0.996));
    #     positive swing goes BACKWARD, so 0 is the natural hang.
    #   shoulder_roll ±0.30 — shoulder sits ~16 mm outboard of the hip
    #     (shoulder_pitch_link y=+0.1002 vs hip_roll y=+0.1165), so a vertical
    #     arm shoves the hand into it: roll=0/elbow=1.50/wrist_roll=∓1.5708
    #     gives 64 penetrating arm<->leg pairs (41 hand<->hip, worst -42.1 mm).
    #     ±0.30 swings it 19.7 deg outboard, +85 mm hull clearance (sweep at
    #     elbow=1.50/wrist_roll=∓1.5708: 0.00 -> -42.1, 0.15 -> +12.1,
    #     0.25 -> +63.8, 0.30 -> +85.1, 0.40 -> +121.5; first clean roll
    #     between 0.10 and 0.15).
    #   elbow 1.50 — elbow zero is already 90 deg flexed (forearm points
    #     forward, ~85 deg from straight-down). Forearm tilt from straight-down
    #     bottoms out at 18.27 deg (elbow=1.4708); 1.50 gives 18.34 deg (0.07
    #     off the minimum), 1.5708 gives 19.12 deg.
    #   wrist_roll ∓1.5708 — this is what decides if the THUMB points at the
    #     thigh. At 0 the thumb points inboard (world (0,-0.94,-0.35)) and the
    #     palm faces backward; ∓pi/2 rolls the palm inboard, thumb forward
    #     (+X) — natural hanging hand, plus +81 mm hip clearance at
    #     roll=0.30/elbow=1.50 (0 -> +0.0043 m, -0.7854 -> +0.0432,
    #     -1.5708 -> +0.0851). Sign follows CARRY_TUCK/NEUTRAL: left
    #     negative, right positive.
    #
    # sim at this exact vector (MuJoCo signed contact distances, margin 0.12,
    # convex-hull geometry):
    #   min arm<->leg   = +0.0851 m  (right_hip_pitch_link x right_wrist_pitch_link)
    #   min arm<->torso = +0.1449 m  (pelvis x left_elbow_link)
    #   hands at (+-0.023, +-0.265, 0.611) m, all 14 joints inside limits.
    # Hulls are fatter than the real shells (torso_link.STL hull 1.77x mesh
    # volume, left_wrist_yaw_link.STL 10.5x), so 85 mm is a LOWER bound — but
    # the two links that set it are near-tight (right_hip_pitch_link 1.08x,
    # right_wrist_pitch_link 1.12x), so take 85 mm as the number.
    "HOME": {
        "left": [0.00, 0.30, 0.00, 1.50, -1.5708, 0.00, 0.00],
        "right": [0.00, -0.30, 0.00, 1.50, 1.5708, 0.00, 0.00],
    },
}


class ArmGuard:
    def __init__(self, cfg):
        self.kp = np.array(cfg["arm_kp"], dtype=np.float32)
        self.kd = np.array(cfg["arm_kd"], dtype=np.float32)
        self.slew_rate = float(cfg["arm_slew_rate"])
        self.slew_min = float(cfg["arm_slew_min_s"])
        self.gate_q = float(cfg["gate_q"])           # our convention -- probe it on the robot
        self.gate_kp = float(cfg["gate_kp"])
        self.gate_kd = float(cfg["gate_kd"])
        self.trail_kp = float(cfg["trailing_kp"])
        self.trail_kd = float(cfg["trailing_kd"])

        self.state = "HOLD_CAPTURED"
        self.current = np.zeros(ARM_SPAN, dtype=np.float32)  # the slewed q
        self.target = np.zeros(ARM_SPAN, dtype=np.float32)
        self._start = np.zeros(ARM_SPAN, dtype=np.float32)
        self._t0 = 0.0
        self._T = 0.0

    # --- little pose helpers ---

    @staticmethod
    def pose_target(name):
        p = ARM_POSES[name]
        return np.concatenate([p["left"], p["right"]]).astype(np.float32)

    # --- the state machine ---

    def capture(self, q, t):
        """Boot: hold the captured arm pose until the first command."""
        self.current = q[L_OFF:L_OFF + ARM_SPAN].copy()
        self.target = self.current.copy()
        self.state = "HOLD_CAPTURED"
        self._T = 0.0

