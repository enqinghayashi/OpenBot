"""obs_builder.py — 516-D GR00T whole-body observation (pure numpy, no SDK).

Exact port of SONIC/decoupled_wbc g1_gear_wbc_policy.compute_observation +
run_mujoco_gear_wbc.py. 516 = 6 frames x 86 dims, concatenated OLDEST-FIRST,
history zero-filled at boot.

86-dim frame layout (verified from source):
    [0:3]    cmd x cmd_scale            (raw joystick [vx, vy, omega] x [2,2,0.5])
    [3]      height_cmd raw              (0.74 m, no scale)
    [4:7]    rpy_cmd                     (fixed [0,0,0] in v1)
    [7:10]   base ang vel x 0.5           (lowstate imu gyroscope)
    [10:13]  projected gravity            (quat_rotate_inverse(quat_wxyz, [0,0,-1]))
    [13:42]  (q - default_padded) x 1.0   (29 motor q, arms zero-pad)
    [42:71]  dq x 0.05                    (29 motor dq)
    [71:86]  prev 15-D action             (policy output from previous tick)

Laptop-testable: no unitree imports, feed synthetic lowstate slices.
"""

import numpy as np

NUM_DOF = 29        # legs 0-11, waist 12-14, arms 15-28
NUM_ACTIONS = 15    # policy output
FRAME_DIM = 86


class ObsBuilder:
