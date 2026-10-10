#!/usr/bin/env python3
"""mj_stance_probe.py — WHY the runtime `stance` verb cannot move the legs.

THE QUESTION
The writer's runtime `stance <deg>` lane shifts BOTH policy-side encodings by
the same delta d (policy_runner.defaults and obs_builder.default_padded, on the
four leg-roll indices 1/5/7/11), on the argument that this is an exact
reparametrisation: the observation is unchanged while the action->target
readout moves by d, so the policy's fixed point translates.

On hardware `stance 4` returned DONE, `status` read `stance=+4.0`, and the
measured hip rolls were IDENTICAL to 0.1 deg (L +0.3, R +0.9) before and after.
The legs did not move at all.

THE ALGEBRA THIS PROBE TESTS
The shipped argument is wrong in one term.  policy_runner:50 is
    target_q = action * action_scale + defaults                      (0.25)
and obs_builder:105 is
    f[13:42] = (q - default_padded) * dof_pos_scale                  (1.0)
`q` is the MEASURED joint angle and does not carry a `+d`.  So shifting both
by d changes the policy's INPUT by -d*1.0 while the readout gains +d.  For a
policy that holds the joint where it is -- action = (q - D)/0.25, i.e.
target_q = q -- the two cancel IDENTICALLY:

    a'      = (q - D - d)/0.25
    target' = (D + d) + 0.25*a' = D + d + q - D - d = q                ==  target

Zero PD torque, zero motion.  And the operator's hand-push is the hardware
proof that the policy IS such a hold: a hand-set 39.86 cm splay survived a
descend, a pick and a retreat, which a restoring hip-roll loop at kp=150 could
not have allowed.

WHAT IS MEASURED, AND THE PRE-REGISTERED PREDICTION FOR EACH
Each mode runs the SAME pinned scene with the SAME real ONNX Balance policy and
the writer's real PD (leg_kp/leg_kd, ctrl = kp*(target-q) - kd*dq + tau).  A
mode is a SUCCESS only if it moves the separation AND the move HOLDS after the
mechanism is removed.  Predictions are registered BEFORE the run:

  control       nothing applied                     d_sep ~ 0      +0.76 cm
  shift-both    the SHIPPED stance mechanism        d_sep ~ 0       +2.95 cm
  shift-policy  pol.defaults only                   creeps out     +2.53 cm
  shift-obs     obs.default_padded only             creeps IN      +1.72 cm
  tau           closed-loop torque on the measured
                hip roll; references untouched      splays + HOLDS  +1.54 cm
  rpy-roll      obs.rpy_cmd[0] (the f[4:7] channel,
                measured LIVE at 0.0285 rad)        no prediction   +0.77 cm

~ THE FIRST RUN OF THIS PROBE FALSIFIED ITS OWN AUTHOR TWICE, and both
corrections are load-bearing:

  1. `shift-both` was predicted to be a structural NO-OP.  It is the
     STRONGEST mover of the six.  The "a hold policy cancels the dual shift
     identically" algebra is wrong -- the policy is compliant, not a hold, so
     it absorbs only part of the -d input perturbation.
  2. `shift-policy` / `shift-obs` were predicted to chase without settling,
     with `shift-obs` moving INWARD.  Both settled, and `shift-obs` moved
     OUTWARD.  The sign reasoning was wrong.

And the control is not flat: the untouched baseline drifts +0.76 cm over the
7 s run, which is the same order as two of the six effects.  Modes are
therefore graded AGAINST THE CONTROL, never against zero.  `rpy-roll` is the
casualty of getting that wrong the first time: at +0.77 cm it is exactly the
control, and the f[4:7] channel -- "LIVE" for base tilt at 0.0285 rad -- does
NOT splay the feet at all.

Run: python3 tools/mj_stance_probe.py --mode all
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
import yaml     # noqa: E402

from obs_builder import ObsBuilder       # noqa: E402
from policy_runner import PolicyRunner   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CFG = os.path.join(PKG, "config", "merge_config.yaml")
XML_DEFAULT = os.path.join(PKG, "sim", "g1_gear_wbc.xml")

# 1/5/7/11: L hipr, L ankr, R hipr, R ankr
# same four the stance lane writes
# OUTWARD: +q[1], -q[7]
# (q[1]=+th, q[7]=-th, th>0)
I_LHIPR, I_LANKR, I_RHIPR, I_RANKR = 1, 5, 7, 11
ANKLE_RATIO = -0.85          # matches STANCE_ANKLE_RATIO

POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

FALL_DROP = 0.12     # z < start-0.12 m = collapse (mj_height_sweep)
MAX_TILT = 0.6       # rad; past squat lean, under topple


def quat_roll_pitch(q_wxyz):
    w, x, y, z = q_wxyz
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch
