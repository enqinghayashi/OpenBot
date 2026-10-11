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


class Rig:
    """The pinned scene + the real Balance policy + the writer's real PD."""

    def __init__(self, xml, onnx_dir, cfg):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = self.model.opt.timestep
        self.policy_every = max(1, int(round(0.02 / self.sim_dt)))

        jnt = {self.model.joint(i).name: i for i in range(self.model.njnt)}
        act = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self.qid = np.zeros(15, dtype=int)
        self.vid = np.zeros(15, dtype=int)
        self.aid = np.full(15, -1, dtype=int)
        for i, nm in enumerate(POLICY_JOINTS):
            j = jnt[nm]
            self.qid[i] = self.model.jnt_qposadr[j]
            self.vid[i] = self.model.jnt_dofadr[j]
            if nm in act:
                self.aid[i] = act[nm]
        if not np.all(self.aid >= 0):
            raise RuntimeError("missing actuators for the 15 policy joints")

        bid = {self.model.body(i).name: i for i in range(self.model.nbody)}
        self.foot_bid = (bid["left_ankle_roll_link"], bid["right_ankle_roll_link"])
        self.pelvis_bid = bid["pelvis"]

        self.obs = ObsBuilder(cfg)
        self.pol = PolicyRunner(cfg, onnx_dir)
        self.kp = np.array(cfg["leg_kp"], dtype=np.float32)
        self.kd = np.array(cfg["leg_kd"], dtype=np.float32)
        self.defaults = np.array(cfg["default_angles"], dtype=np.float32)

        self.arm_act = [i for i in range(self.model.nu) if i not in self.aid.tolist()]
        self.arm_kp = np.full(len(self.arm_act), 100.0)
        self.arm_kd = np.full(len(self.arm_act), 0.5)

        # boot encodings the lane snapshots
        # 2 separate copies
        self.base_pol = self.pol.defaults.copy()
        self.base_obs = self.obs.default_padded.copy()

    # --- measuring the feet ----------------------------------------------
    def separation(self):
        """(world 3-D, pelvis-frame 3-D, pelvis-frame lateral dy) foot gap.

        feet_dist.py measures in the PELVIS frame, so the pelvis-frame numbers
        are the ones comparable with the operator's on-robot readings.
        """
        pl = self.data.xpos[self.foot_bid[0]].copy()
        pr = self.data.xpos[self.foot_bid[1]].copy()
        Rp = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        vl, vr = Rp.T @ pl, Rp.T @ pr
        return (float(np.linalg.norm(pl - pr)),
                float(np.linalg.norm(vl - vr)),
                float(abs(vl[1] - vr[1])))

    def hip_rolls(self):
        q = self.data.qpos[self.qid]
        return float(np.degrees(q[I_LHIPR])), float(np.degrees(q[I_RHIPR]))

    def reset_run(self, start_height):
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        d.qpos[self.qid] = self.defaults
        d.qpos[2] = start_height
        mujoco.mj_forward(self.model, d)
        self.obs.reset()
        self.obs.height_cmd = start_height
        self.obs.rpy_cmd = np.zeros(3, dtype=np.float32)
        self.pol.action[:] = 0.0
        self.pol.defaults = self.base_pol.copy()
        self.obs.default_padded = self.base_obs.copy()


def stance_delta(deg):
    """The writer's exact delta vectors for a `stance <deg>` command."""
    rad = np.radians(deg)
    hip = rad
    ank = ANKLE_RATIO * rad
    d_pol = np.zeros(15, dtype=np.float32)
    d_pol[I_LHIPR] = +hip
    d_pol[I_RHIPR] = -hip
    d_pol[I_LANKR] = +ank
    d_pol[I_RANKR] = -ank
    d_obs = np.zeros(29, dtype=np.float32)
    d_obs[0:15] = d_pol
    return d_pol, d_obs


def run_mode(rig, mode, deg, tau_max, servo_kp, t_settle, t_probe, t_post,
             rpy_val):
    """settle -> probe (mechanism ON) -> post (mechanism OFF). -> result dict"""
    r = rig
    start_height = 0.74
    r.reset_run(start_height)

    ramp_t = 1.0                       # ramp long done
    z0 = None
    d_pol, d_obs = stance_delta(deg)
    applied = False
    tau_on = False
    hold_ref = None                    # hip roll at start

    dt = r.sim_dt
    total = t_settle + t_probe + t_post
    n = int(round(total / dt))
    sep_t, sep_lat, hipL_t, hipR_t, t_t = ([] for _ in range(5))
    fell, why = False, ""
    prev = np.zeros(15, dtype=np.float32)

    for step in range(n):
        t = step * dt
        phase = ("settle" if t < t_settle
                 else "probe" if t < t_settle + t_probe else "post")

        # ---- mechanism ON at probe boundary
        # writer does same
        if phase != "settle" and not applied:
            applied = True
            if mode == "shift-both":
                r.pol.defaults = r.base_pol + d_pol
                r.obs.default_padded = r.base_obs + d_obs
            elif mode == "shift-policy":
                r.pol.defaults = r.base_pol + d_pol
            elif mode == "shift-obs":
                r.obs.default_padded = r.base_obs + d_obs
            elif mode == "tau":
                tau_on = True
                hold_ref = np.array(r.data.qpos[r.qid])[[I_LHIPR, I_RHIPR]]
            elif mode == "rpy-roll":
                r.obs.rpy_cmd = np.array([rpy_val, 0.0, 0.0], dtype=np.float32)
        # ---- OFF at the post boundary
        if phase == "post" and tau_on:
            tau_on = False
        if phase == "post" and mode in ("shift-both", "shift-policy", "shift-obs"):
            r.pol.defaults = r.base_pol.copy()
            r.obs.default_padded = r.base_obs.copy()
        if phase == "post" and mode == "rpy-roll":
            r.obs.rpy_cmd = np.zeros(3, dtype=np.float32)

        if step % r.policy_every == 0:
            q29 = r.data.qpos[7:36].astype(np.float32)
            dq29 = r.data.qvel[6:35].astype(np.float32)
            quat = r.data.qpos[3:7].astype(np.float32)
            gyro = r.data.qvel[3:6].astype(np.float32)
            o = r.obs.push(q29, dq29, quat, gyro, (0.0, 0.0, 0.0), prev)
            target = r.pol.infer(o, (0.0, 0.0, 0.0))
            prev = r.pol.action.copy()
        else:
            target = r.pol.target_q

        q = r.data.qpos[r.qid]
        dq = r.data.qvel[r.vid]
        ctrl = r.kp * (target - q) - r.kd * dq

        # ---- torque servo
        # on the measured hip roll
        if mode == "tau" and tau_on:
            wantL = hold_ref[0] + np.radians(deg)
            wantR = hold_ref[1] - np.radians(deg)
            errL = wantL - r.data.qpos[r.qid[I_LHIPR]]
            errR = wantR - r.data.qpos[r.qid[I_RHIPR]]
            ctrl[I_LHIPR] += float(np.clip(servo_kp * errL, -tau_max, tau_max))
            ctrl[I_RHIPR] += float(np.clip(servo_kp * errR, -tau_max, tau_max))

        r.data.ctrl[r.aid] = ctrl
        qa = r.data.qpos[7:36][15:]
        dqa = r.data.qvel[6:35][15:]
        r.data.ctrl[r.arm_act] = r.arm_kp * (0.0 - qa) - r.arm_kd * dqa
        mujoco.mj_step(r.model, r.data)

        zw = float(r.data.qpos[2])
        roll, pitch = quat_roll_pitch(r.data.qpos[3:7])
        if z0 is None and t >= t_settle * 0.5:
            z0 = zw
        if z0 is not None and (zw < z0 - FALL_DROP or max(abs(roll), abs(pitch)) > MAX_TILT):
            fell, why = True, (f"z {zw:.3f} (start {z0:.3f})" if zw < z0 - FALL_DROP
                               else f"tilt {max(abs(roll), abs(pitch)):.3f} rad")
            break

        sepw, sepp, lat = r.separation()
        hl, hr = r.hip_rolls()
        t_t.append(t); sep_t.append(sepp); sep_lat.append(lat)
        hipL_t.append(hl); hipR_t.append(hr)

    if not sep_t:
        return {"mode": mode, "fell": True, "why": why,
                "sep_settle": None, "sep_probe": None, "sep_post": None}

    def at(tt, arr):
        i = int(np.argmin(np.abs(np.array(t_t) - tt)))
        return float(arr[i])

    out = {
        "mode": mode,
        "deg": deg,
        "fell": fell, "why": why,
        "sep_settle": at(t_settle - 0.02, sep_t),
        "sep_probe": at(t_settle + t_probe - 0.02, sep_t),
        "sep_post": at(total - 0.02, sep_t),
        "lat_settle": at(t_settle - 0.02, sep_lat),
        "lat_probe": at(t_settle + t_probe - 0.02, sep_lat),
        "lat_post": at(total - 0.02, sep_lat),
        "hipL_settle": at(t_settle - 0.02, hipL_t),
        "hipL_probe": at(t_settle + t_probe - 0.02, hipL_t),
        "hipL_post": at(total - 0.02, hipL_t),
        "hipR_settle": at(t_settle - 0.02, hipR_t),
        "hipR_probe": at(t_settle + t_probe - 0.02, hipR_t),
        "hipR_post": at(total - 0.02, hipR_t),
        "sep_peak": float(np.max(sep_t)),
        "sep_min": float(np.min(sep_t)),
    }
    out["d_probe"] = out["sep_probe"] - out["sep_settle"]
    out["d_post"] = out["sep_post"] - out["sep_settle"]
    out["drift_after"] = out["sep_post"] - out["sep_probe"]
    return out


PREDICTION = {
    "control":      "d_sep ~ 0 (baseline)",
    "shift-both":   "d_sep ~ 0 -- the shipped mechanism is a structural NO-OP",
    "shift-policy": "creeps OUTWARD and does not settle (constant-torque chase)",
    "shift-obs":    "creeps INWARD and does not settle (constant-torque chase)",
    "tau":          "splays during the pulse, then HOLDS (drift_after ~ 0)",
    "rpy-roll":     "measurement only -- the f[4:7] channel is live (0.0285 rad)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.environ.get("MERGE_MUJOCO_XML", XML_DEFAULT))
    ap.add_argument("--onnx-dir", default=os.path.join(PKG, "policy"))
    ap.add_argument("--mode", nargs="+", default=["all"])
    ap.add_argument("--deg", type=float, default=4.0,
                    help="splay commanded, degrees (the value the operator sent)")
    ap.add_argument("--tau-max", type=float, default=18.0, help="N.m torque cap")
    ap.add_argument("--servo-kp", type=float, default=200.0, help="N.m/rad")
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--probe-s", type=float, default=4.0)
    ap.add_argument("--post-s", type=float, default=3.0)
    ap.add_argument("--rpy-val", type=float, default=0.15, help="rad, roll command")
    ap.add_argument("--out", default=os.path.join(HERE, "out_stance_probe"))
    args = ap.parse_args()

    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)

    modes = (list(PREDICTION) if args.mode == ["all"] else args.mode)
    bad = [m for m in modes if m not in PREDICTION]
    if bad:
        ap.error(f"unknown mode(s): {bad}")

    rig = Rig(args.xml, args.onnx_dir, cfg)
    zg = float(cfg["default_angles"][3])
    print("=" * 78)
    print("PINNED SCENE + REAL ONNX BALANCE POLICY + THE WRITER'S REAL PD")
    print(f"  scene {args.xml}")
    print(f"  sim_dt={rig.sim_dt}  policy {1.0/(rig.policy_every*rig.sim_dt):.0f} Hz")
    print(f"  leg_kp[hip_roll]={rig.kp[1]:.0f} N.m/rad   N.m per {args.deg} deg = "
          f"{rig.kp[1]*np.radians(args.deg):.2f}")
    print(f"  action_scale={cfg['action_scale']}  dof_pos_scale={cfg['dof_pos_scale']}"
          f"  -> a default_angles shift of d is cancelled by the obs term -d*s")
    print(f"  knee default {zg:+.3f} rad; phases settle {args.settle_s}s / "
          f"probe {args.probe_s}s / post {args.post_s}s")
    print("=" * 78)

    rows = []
    for m in modes:
        print()
        print(f"--- {m} " + "-" * (70 - len(m)))
        print(f"  PRE-REGISTERED PREDICTION: {PREDICTION[m]}")
        r = run_mode(rig, m, args.deg, args.tau_max, args.servo_kp,
                     args.settle_s, args.probe_s, args.post_s, args.rpy_val)
        rows.append(r)
        if r.get("sep_settle") is None:
            print(f"  FELL before any sample: {r['why']}")
            continue
        print(f"  separation (pelvis frame, ankle-to-ankle)")
        print(f"    at settle end : {r['sep_settle']*100:7.2f} cm")
        print(f"    at probe end  : {r['sep_probe']*100:7.2f} cm   "
              f"(d = {r['d_probe']*100:+6.2f} cm)")
        print(f"    at post end   : {r['sep_post']*100:7.2f} cm   "
              f"(d = {r['d_post']*100:+6.2f} cm)")
        print(f"    drift after the mechanism was removed: "
              f"{r['drift_after']*100:+6.2f} cm")
        print(f"    peak {r['sep_peak']*100:.2f} cm   min {r['sep_min']*100:.2f} cm")
        print(f"  hip roll (deg)  L {r['hipL_settle']:+.2f} -> {r['hipL_probe']:+.2f} "
              f"-> {r['hipL_post']:+.2f}   |   R {r['hipR_settle']:+.2f} -> "
              f"{r['hipR_probe']:+.2f} -> {r['hipR_post']:+.2f}")
        print(f"  fell: {'YES -- ' + r['why'] if r['fell'] else 'no'}")

    print()
    print("=" * 78)
    # control not flat: +0.76 cm / 7 s
    # grade against CONTROL, never zero
    # baseline alone moves +0.8 cm
    # first run: rpy-roll +0.77 -> NO-OP,
    # tau +1.54 -> success on +0.76
    ctl = next((r for r in rows if r["mode"] == "control"
                and r.get("sep_settle") is not None), None)
    c_post = ctl["d_post"] if ctl else 0.0
    c_drift = ctl["drift_after"] if ctl else 0.0
    print(f"SUMMARY  (commanded {args.deg:+.1f} deg splay; predictions are above)")
    if ctl:
        print(f"  CONTROL baseline: d_post {c_post*100:+.2f} cm, drift "
              f"{c_drift*100:+.2f} cm -- every mode is graded AGAINST this, "
              f"never against zero")
    print(f"  {'mode':<13} {'d_probe':>9} {'d_post':>9} {'vs_ctrl':>9} "
          f"{'drift_after':>12} {'hold?':>6}  fell")
    verdict = {}
    for r in rows:
        if r.get("sep_settle") is None:
            print(f"  {r['mode']:<13} {'--':>9} {'--':>9} {'--':>9} {'--':>12} "
                  f"{'--':>6}  YES")
            verdict[r["mode"]] = "FELL"
            continue
        vs_ctrl = r["d_post"] - c_post
        moved = abs(vs_ctrl) > 0.01               # >1 cm over control
        held = abs(r["drift_after"] - c_drift) < 0.01
        print(f"  {r['mode']:<13} {r['d_probe']*100:+8.2f} {r['d_post']*100:+8.2f} "
              f"{vs_ctrl*100:+8.2f} {r['drift_after']*100:+11.2f} "
              f"{'YES' if held else 'NO':>6}  {'YES' if r['fell'] else 'no'}")
        if r["mode"] == "control":
            verdict[r["mode"]] = "BASELINE"
        else:
            verdict[r["mode"]] = ("SPLAYS-AND-HOLDS" if (moved and held)
                                  else "MOVES-BUT-DRIFTS" if moved
                                  else "NO-OP")

    print()
    print("  verdicts (control-referenced), against the registered predictions:")
    for m in modes:
        v = verdict.get(m, "?")
        print(f"    {m:<13} -> {v}")

    with open(os.path.join(args.out, "metrics_stance_probe.json"), "w") as f:
        json.dump({"mode": modes, "deg": args.deg, "tau_max": args.tau_max,
                   "servo_kp": args.servo_kp, "rows": rows,
                   "verdict": verdict,
                   "claim_scope": (
                       "PINNED MuJoCo scene + the real deployed ONNX Balance "
                       "policy + the writer's real leg_kp/leg_kd PD.  This is a "
                       "SIMULATION, not the robot.  A mode is a SUCCESS only if "
                       "it moves the separation AND the move survives the "
                       "mechanism being removed.")},
                  f, indent=1)
    print(f"\nwrote {os.path.join(args.out, 'metrics_stance_probe.json')}")
    return 0
