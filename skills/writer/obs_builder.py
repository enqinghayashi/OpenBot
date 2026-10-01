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
    def __init__(self, cfg):
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.ang_vel_scale = float(cfg["ang_vel_scale"])
        self.dof_pos_scale = float(cfg["dof_pos_scale"])
        self.dof_vel_scale = float(cfg["dof_vel_scale"])
        self.height_cmd = float(cfg["height_cmd"])
        # height band, clamped; missing keys -> height_cmd,
        # so the setter can't move it (old-config fallback)
        self.height_min = float(cfg.get("height_cmd_min", cfg["height_cmd"]))
        self.height_max = float(cfg.get("height_cmd_max", cfg["height_cmd"]))
        self.rpy_cmd = np.array(cfg["rpy_cmd"], dtype=np.float32)
        self.history_len = int(cfg["obs_history_len"])

        defaults = np.array(cfg["default_angles"], dtype=np.float32)  # 15: legs + waist
        assert defaults.shape == (NUM_ACTIONS,)
        self.default_padded = np.concatenate(
            [defaults, np.zeros(NUM_DOF - NUM_ACTIONS, dtype=np.float32)])

        self._frame = np.zeros(FRAME_DIM, dtype=np.float32)
        self.obs = np.zeros(self.history_len * FRAME_DIM, dtype=np.float32)
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.float32)  # feeds obs[71:86]

