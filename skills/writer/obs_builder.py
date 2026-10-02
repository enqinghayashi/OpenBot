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

    def set_height(self, value):
        """Runtime height-reference setter — clamped, allocation-free.

        Called on the writer's 250 Hz loop immediately before the policy push,
        so it does nothing but compare-and-assign: no numpy allocation, no
        logging, and above all NO assert.  An AssertionError raised here would
        bypass the writer's RuntimeError-only hot-path handlers and abort the
        crash-safe shutdown tail (the known incident class) — the band is a
        clamp, never an assert.

        `not (x >= lo)` also maps NaN to the floor, and +/-inf to the nearest
        bound, so a non-finite reference can never poison obs frame f[3].
        The band is the configured safe range; rate limiting (height_slew_rate)
        is the caller's job — this setter only clamps.
        """
        if not (value >= self.height_min):
            value = self.height_min
        elif not (value <= self.height_max):
            value = self.height_max
        self.height_cmd = value

    # --- isaac/legged-gym helpers ---

    @staticmethod
    def quat_rotate_inverse(q_wxyz, v):
        """Rotate vector v into the body frame: q^-1 (0,v) q. q = (w,x,y,z)."""
        q = np.asarray(q_wxyz, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        else:
            qw, qx, qy, qz = q / norm
        qvec = np.array([qx, qy, qz])
        v = np.asarray(v, dtype=np.float64)
        # a-b+c: v(2w^2-1) - 2w(qvec x v) + 2(qvec.v)qvec
        return (v * (2.0 * qw * qw - 1.0)
                - 2.0 * qw * np.cross(qvec, v)
                + 2.0 * qvec * np.dot(qvec, v)).astype(np.float32)

    def projected_gravity(self, quat_wxyz):
        return self.quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0]))

    # --- frames ---

    def build_frame(self, q, dq, quat_wxyz, gyro, cmd, prev_action=None):
        """86-dim frame from lowstate slices + raw rc cmd. q/dq are 35-length
        motor arrays (only [0:29] used)."""
        f = self._frame
        f[0:3] = np.asarray(cmd, dtype=np.float32) * self.cmd_scale
        f[3] = self.height_cmd
        f[4:7] = self.rpy_cmd
        f[7:10] = np.asarray(gyro, dtype=np.float32) * self.ang_vel_scale
        f[10:13] = self.projected_gravity(quat_wxyz)
        f[13:42] = (np.asarray(q, dtype=np.float32)[0:NUM_DOF] - self.default_padded) \
            * self.dof_pos_scale
        f[42:71] = np.asarray(dq, dtype=np.float32)[0:NUM_DOF] * self.dof_vel_scale
        f[71:86] = self.prev_action if prev_action is None else prev_action
        return f

