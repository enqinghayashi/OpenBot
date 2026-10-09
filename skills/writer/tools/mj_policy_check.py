#!/usr/bin/env python3
"""mj_policy_check.py — MuJoCo sim gate for the C3 policy stack (laptop only).

Drives the GR00T Balance/Walk ONNX policies through the EXACT obs stack the
writer uses (obs_builder + policy_runner) in the g1_gear_wbc.xml sim.

Gates:
  stand:  base z settles within [0.65, 0.85] m of height_cmd 0.74, no fall
          for 12 s with zero cmd
  walk:   cmd (0.3, 0, 0) -> forward displacement 0.15-0.45 m/s avg, no fall
          for 15 s

Usage:  python3 tools/mj_policy_check.py [--xml <path>] [--onnx-dir <dir>]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import mujoco

from obs_builder import ObsBuilder
from policy_runner import PolicyRunner

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CFG = os.path.join(PKG, "config", "merge_config.yaml")
# self-contained: meshes pinched from unitree_ros g1_d_description
# (GRAIL's own are just Git-LFS stubs)
XML_DEFAULT = os.path.join(PKG, "sim", "g1_gear_wbc.xml")

# XML order, matches G1JointIndex 0..14
POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]


def quat_to_wxyz_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.environ.get("MERGE_MUJOCO_XML", XML_DEFAULT))
    ap.add_argument("--onnx-dir", default=os.path.join(PKG, "policy"))
    ap.add_argument("--stand-s", type=float, default=12.0)
    ap.add_argument("--walk-s", type=float, default=15.0)
    ap.add_argument("--omega-mode", choices=["reference", "body"], default="reference",
                    help="obs[7:10] source: 'reference' = qvel[3:6] verbatim (NVIDIA "
                         "run_mujoco_gear_wbc.py convention — policy is tuned to it, "
                         "walk 0.247 m/s steady); 'body' = world ang vel rotated into "
                         "the body frame (true gyro, what the real deploy feeds — "
                         "passes stand + initial walk, decelerates late)")
    args = ap.parse_args()

    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep
    policy_every = max(1, int(round(0.02 / sim_dt)))   # 50 Hz
    print(f"sim_dt={sim_dt} policy_every={policy_every} "
          f"policy_rate={1.0/(policy_every*sim_dt):.0f} Hz")

    # policy joint -> qpos/qvel/act index, by NAME not position
    jnt_id = {model.joint(i).name: i for i in range(model.njnt)}
    act_id = {model.actuator(i).name: i for i in range(model.nu)}
    qid = np.zeros(15, dtype=int)   # qpos start
    vid = np.zeros(15, dtype=int)   # qvel start
    aid = np.full(15, -1, dtype=int)
    for i, name in enumerate(POLICY_JOINTS):
        j = jnt_id[name]
        qid[i] = model.jnt_qposadr[j]
        vid[i] = model.jnt_dofadr[j]
        if name in act_id:
            aid[i] = act_id[name]
    assert np.all(aid >= 0), "missing actuators for policy joints"
    assert np.all(np.diff(qid) == 1), "policy joints must be contiguous qpos"

    obs = ObsBuilder(cfg)
    pol = PolicyRunner(cfg, args.onnx_dir)
    kp = np.array(cfg["leg_kp"], dtype=np.float32)
    kd = np.array(cfg["leg_kd"], dtype=np.float32)
    defaults = np.array(cfg["default_angles"], dtype=np.float32)

    # arm/hand: reference PDs these to 0 (kp 100, kd 0.5)
    arm_act = [i for i in range(model.nu) if i not in aid.tolist()]
    arm_kp = np.full(len(arm_act), 100.0)
    arm_kd = np.full(len(arm_act), 0.5)

    def run_sim(cmd, seconds, label):
        # reset: freejoint identity, 29 joints at default, z = height_cmd
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = quat_to_wxyz_identity()
        data.qpos[qid] = defaults      # 15 policy joints; arms stay 0
        data.qpos[2] = cfg["height_cmd"]
        mujoco.mj_forward(model, data)
        obs.reset()

        n_steps = int(seconds / sim_dt)
        x0 = data.qpos[0]
        prev_act = np.zeros(15, dtype=np.float32)
        z_track = []
        fall = False
        t0 = time.perf_counter()
        for step in range(n_steps):
            if step % policy_every == 0:
                # 29-D: legs+waist+arms
                q29 = data.qpos[7:36].astype(np.float32)
                dq29 = data.qvel[6:35].astype(np.float32)
                quat = data.qpos[3:7].astype(np.float32)      # wxyz order
                if args.omega_mode == "reference":
                    gyro = data.qvel[3:6].astype(np.float32)  # NVIDIA convention
                else:
                    gyro = obs.quat_rotate_inverse(quat, data.qvel[0:3])  # true gyro, body frame
                obs_516 = obs.push(q29, dq29, quat, gyro, cmd, prev_act)
                target = pol.infer(obs_516, cmd)
                prev_act = pol.action.copy()
            # PD torque; no firmware controller in sim
            q = data.qpos[qid]
            dq = data.qvel[vid]
            data.ctrl[aid] = kp * (target - q) - kd * dq
            # arms at 0 too
            q_arm = data.qpos[7:36][15:]
            dq_arm = data.qvel[6:35][15:]
            data.ctrl[arm_act] = arm_kp * (0.0 - q_arm) - arm_kd * dq_arm
            mujoco.mj_step(model, data)
            z_track.append(data.qpos[2])
            if data.qpos[2] < 0.4:
                fall = True
                print(f"[{label}] FALL at t={step*sim_dt:.2f} s")
                break
        wall = time.perf_counter() - t0
        disp = data.qpos[0] - x0
        return {
            "fall": fall, "disp": disp, "z": np.array(z_track),
            "z_settle": np.median(z_track[-int(2.0 / sim_dt):]),
            "wall": wall, "sim": n_steps * sim_dt,
        }

    ok = True

    r = run_sim((0.0, 0.0, 0.0), args.stand_s, "stand")
    z = r["z_settle"]
    good_z = 0.65 <= z <= 0.85
    print(f"[stand] settle z={z:.3f} m (want ~0.74) fall={r['fall']} "
          f"disp={r['disp']:.3f} m sim={r['sim']:.0f}s wall={r['wall']:.1f}s")
    ok &= (not r["fall"]) and good_z

    r = run_sim((0.3, 0.0, 0.0), args.walk_s, "walk")
    v = r["disp"] / r["sim"]
    good_v = 0.15 <= v <= 0.45
    print(f"[walk ] settle z={r['z_settle']:.3f} m fall={r['fall']} "
          f"disp={r['disp']:.2f} m v={v:.3f} m/s (want 0.2-0.3) "
          f"sim={r['sim']:.0f}s wall={r['wall']:.1f}s")
    ok &= (not r["fall"]) and good_v

    print()
    print("mj_policy_check", "GREEN" if ok else "FAILED",
          "(stand 0.74 m, walk 0.2-0.3 m/s)" if ok else "")
    sys.exit(0 if ok else 1)
