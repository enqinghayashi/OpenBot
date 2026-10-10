#!/usr/bin/env python3
"""mj_sonic_track.py — staged closed-loop SONIC stack test in the OpenBot MuJoCo scene.

Runs the FULL SONIC stack (planner -> encoder -> decoder tracker) closed-loop in
skills/writer/sim/g1_gear_wbc.xml (loaded READ-ONLY; byte-pinned md5
8ffca967c299400100d80d2ce40a917f) and reports whether the robot stands.

Three staged tests, one JSON line each (A first, and non-negotiable):
  A) TRACKER-ONLY  : constant standing reference (SONIC default_angles), no planner. 6 s
  B) PLANNER IDLE  : planner mode 0, then tracker.                                    6 s
  C) PLANNER WALK  : planner mode 2, target_vel 1.0, then tracker.                    8 s
A result is only interpretable if A stands.

ACTUATION MODEL — a 50 Hz POSITION setpoint on a 500 Hz PD law:
  The reference deploy never computes a joint torque. Its 50 Hz Control thread publishes
  a position setpoint (q_target, kp, kd) to Unitree's motor controllers, which evaluate the
  PD law themselves, continuously, at ~1 kHz. This harness reproduces that: `q_target` is
  refreshed on the 50 Hz control tick, and the identical PD law is evaluated on EVERY
  500 Hz sim step from that setpoint. Holding the TORQUE across the intervening sim steps
  (a zero-order hold) instead of the SETPOINT is a sim-fidelity defect that makes the loop
  diverge for every reference; see the comment at the control seam for the evidence.

FALL CRITERION — NOT-FALLEN test, not a positive standing test:
  fell = (root tilt from vertical > 45 deg) OR (root z < 0.35)

HONESTY NOTE on that criterion: the nominal standing height in this scene is
0.75 m, so 0.35 is ~53% of body height — a robot collapsed into an upright
crouch/kneel with root z anywhere in [0.35, 0.75] and torso tilt < 45 deg is NOT
flagged. `CONTROL_A_STANDS` is the final line's string because the task spec
mandates it verbatim; read it as "A did not fall". The thresholds are the spec's
and are deliberately NOT tuned here (tuning a fall threshold is the single worst
outcome for this task). The raw z_min / max_tilt_deg numbers in each JSON record,
not the boolean label, are the trustworthy evidence.

The exit status is a machine-readable gate in addition to the printed lines:
0 only when A stands AND B stands AND C walks, 1 otherwise (see main()).
A alone is the non-negotiable precondition; B and C are now meaningful, so they gate too.

===============================================================================
CONSTANTS — imported from skills/sonic/sonic_core.py (SINGLE SOURCE OF TRUTH)
  Originally copied verbatim from the SONIC reference deploy header
  /home/leonlin/Unitree/GRAIL/imports/SONIC/gear_sonic_deploy/src/g1/
      g1_deploy_onnx_ref/include/policy_parameters.hpp
These are SONIC OWN gains. NOT the GR00T / merge_config gains — using OpenBot
leg_kp (150/200/40) or its default_angles here is a DEFECT.

That block was previously duplicated in this file. It now lives only in
sonic_core, because the writer seam and the satellite consume the same tables and
a drifted copy would silently invalidate the closed-loop proof below. The names
are re-exported unchanged; test_sonic_equivalence.py T1 asserts exact
np.array_equal parity for every table against sonic_core.

policy_parameters.hpp:100-101  (isaaclab order in mujoco index)
  const std::array<int, 29> isaaclab_to_mujoco = {0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2,  5, 8,
                                                  11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};

policy_parameters.hpp:103-104  (mujoco order in isaaclab index)
  const std::array<int, 29> mujoco_to_isaaclab = {0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
                                                  16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

policy_parameters.hpp:109-139  g1_action_scale[i] = 0.25 * EFFORT_LIMIT_x / STIFFNESS_x   (muJoCo order)
policy_parameters.hpp:143-173  kps[i]             = STIFFNESS_x  (ankles/waist-roll/pitch x2)
policy_parameters.hpp:177-207  kds[i]             = DAMPING_x    (ankles/waist-roll/pitch x2)
policy_parameters.hpp:210-240  default_angles     = {-0.312,0,0,0.669,-0.363,0,  -0.312,0,0,0.669,-0.363,0,
                                                     0,0,0,  0.2,0.2,0,0.6,0,0,0,  0.2,-0.2,0,0.6,0,0,0}
policy_parameters.hpp:40-65    ARMATURE_{5020,7520_14,7520_22,4010}
                               EFFORT_LIMIT_{5020,7520_14,7520_22,4010} = 25, 88, 139, 5
policy_parameters.hpp:46-47    NATURAL_FREQ = 10*2*pi, DAMPING_RATIO = 2
                               STIFFNESS_x = ARMATURE_x * NATURAL_FREQ**2
                               DAMPING_x   = 2 * DAMPING_RATIO * ARMATURE_x * NATURAL_FREQ

Usage: python3 tools/mj_sonic_track.py [--xml PATH] [--models DIR] [--dump-states PATH]

--dump-states writes ONE .npz holding every control tick of all three tests
(tagged by test_index) for the offline equivalence test T2. Off by default; it
does not alter any control path.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
# sonic_core in skills/sonic
# imported top-level
SONIC_DIR = os.path.join(os.path.dirname(PKG), "sonic")
sys.path.insert(0, SONIC_DIR)

XML_DEFAULT = os.path.join(PKG, "sim", "g1_gear_wbc.xml")
MODELS_DEFAULT = "/home/leonlin/Unitree/planner_models"

# ---------------------------------------------------------------------------
# SONIC constants, imported
# one copy: no drift
# T1: exact np.array_equal parity
# ---------------------------------------------------------------------------
from sonic_core import (  # noqa: E402
    NUM_MOTOR, QPOS_DIM, SIM_DT, CONTROL_EVERY, PLANNER_EVERY, PLANNER_HZ,
    CONTROL_HZ, DEFAULT_HEIGHT,
    ARMATURE_5020, ARMATURE_7520_14, ARMATURE_7520_22, ARMATURE_4010,
    NATURAL_FREQ, DAMPING_RATIO,
    EFFORT_LIMIT_5020, EFFORT_LIMIT_7520_14, EFFORT_LIMIT_7520_22,
    EFFORT_LIMIT_4010,
    STIFFNESS_5020, STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_4010,
    DAMPING_5020, DAMPING_7520_14, DAMPING_7520_22, DAMPING_4010,
    ISAC_TO_MUJOCO, MUJOCO_TO_ISAC, DEFAULT_ANGLES_MUJOCO,
    KP_MUJOCO, KD_MUJOCO, ACTION_SCALE_MUJOCO, ALLOWED_PRED_NUM_TOKENS,
    ENC_MODE_OFF, ENC_JPOS_OFF, ENC_JVEL_OFF, ENC_ANCHOR_OFF, ENC_DIM,
    DEC_TOKEN_OFF, DEC_GYRO_OFF, DEC_JPOS_OFF, DEC_JVEL_OFF, DEC_ACT_OFF,
    DEC_GRAV_OFF, DEC_DIM, ENC_STEP, HIST_LEN, TILT_LIMIT_DEG, Z_FALL,
    quat_mul, quat_conj, quat_normalize, quat_to_mat, quat_rotate_inverse,
    quat_slerp, tilt_deg,
    Motion, make_standing_motion, resample_30_to_50, SonicStack,
    build_planner_context, anchor_orientation_6d, build_encoder_obs,
    build_decoder_obs,
)


# ---------------------------------------------------------------------------
# the sim loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML_DEFAULT)
    ap.add_argument("--models", default=MODELS_DEFAULT)
    ap.add_argument("--dump-states", default=None, metavar="PATH",
                    help="Write a replayable state dump to PATH (.npz) for the "
                         "offline equivalence test. Default off. "
                         "ONE npz for the whole run (all three tests, tagged by "
                         "'test_index'), not one file per tick: T2 replays the "
                         "ticks in order and 400+ tiny files would be pure "
                         "overhead. Nothing else about the run changes.")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep
    assert abs(sim_dt - SIM_DT) < 1e-12, f"sim dt {sim_dt} != {SIM_DT}"
    print(f"sim_dt={sim_dt} nq={model.nq} nv={model.nv} nu={model.nu} "
          f"control_hz={CONTROL_HZ:.0f} planner_hz={CONTROL_HZ * CONTROL_EVERY / PLANNER_EVERY:.0f}")

    jnt_id = {model.joint(i).name: i for i in range(model.njnt)}
    act_id = {model.actuator(i).name: i for i in range(model.nu)}
    # actuator = joint i
    # string order, lower body
    jnames = [model.joint(i).name for i in range(1, model.njnt)]
    aid = np.array([act_id[n] for n in jnames], dtype=np.int64)
    assert np.array_equal(aid, np.arange(NUM_MOTOR)), "actuator order != joint order"
    qadr = np.array([model.jnt_qposadr[jnt_id[n]] for n in jnames], dtype=np.int64)
    vadr = np.array([model.jnt_dofadr[jnt_id[n]] for n in jnames], dtype=np.int64)
    assert np.array_equal(qadr, np.arange(7, 36)), "joint qpos not contiguous 7..35"
    assert np.array_equal(vadr, np.arange(6, 35)), "joint qvel not contiguous 6..34"

    stack = SonicStack(args.models)

    # ctrlrange [0,0] == unlimited
    assert np.all(model.actuator_ctrlrange == 0.0), "expected ctrlrange [0,0]"

    def run_test(label, mode, target_vel, seconds, use_planner):
        # replay records: ~1.1M floats
        # --dump-states writes them
        dump = []
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = 0.0
        data.qpos[2] = DEFAULT_HEIGHT
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[7:36] = DEFAULT_ANGLES_MUJOCO
        mujoco.mj_forward(model, data)

        n_steps = int(round(seconds / sim_dt))
        movement_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        facing_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        motion = make_standing_motion()
        cur_frame = 0
        stack.n_planner_inf = 0
        n_control = 0
        first_plan = True

        prev_action = np.zeros(NUM_MOTOR, dtype=np.float64)   # isaaclab order
        # PD setpoint, 50 Hz refresh
        # PD law every sim step
        q_target = DEFAULT_ANGLES_MUJOCO.copy()
        hist = None
        z_hist, x_hist, t_hist, tilt_hist = [], [], [], []
        fell = False
        n_planner_ticks = 0

        for step in range(n_steps):
            if use_planner and step % PLANNER_EVERY == 0:
                if first_plan:
                    ctx = np.zeros((4, QPOS_DIM), dtype=np.float64)
                    ctx[:, 2] = DEFAULT_HEIGHT
                    ctx[:, 3] = 1.0
                    ctx[:, 7:36] = DEFAULT_ANGLES_MUJOCO
                else:
                    ctx = build_planner_context(motion, cur_frame + 2)
                q30, n30 = stack.plan(ctx, mode, target_vel, movement_dir,
                                      facing_dir, DEFAULT_HEIGHT)
                new_motion = resample_30_to_50(q30, n30)
                if new_motion is not None:
                    # known deviation:
                    # hard cut to frame 0
                    # reference splices:
                    # rebase + crossfade
                    # blend_num_frames = 8
                    # (g1_deploy_onnx_ref.cpp:3189-3288)
                    # then current_frame_ = 0 (cpp:3288)
                    # blend skipped, reset kept
                    # B/C look-ahead may jump phase
                    # each 0.1 s replan
                    motion = new_motion
                    cur_frame = 0
                first_plan = False
                n_planner_ticks += 1

            if step % CONTROL_EVERY == 0:
                q_mj = data.qpos[qadr].astype(np.float64)
                dq_mj = data.qvel[vadr].astype(np.float64)
                base_quat = data.qpos[3:7].astype(np.float64)
                gyro = data.qvel[3:6].astype(np.float64)   # angular vel, body
                grav = quat_rotate_inverse(base_quat, [0.0, 0.0, -1.0])

                cur_val = {
                    "gyro": gyro,
                    "jpos": q_mj[MUJOCO_TO_ISAC] - DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAC],
                    "jvel": dq_mj[MUJOCO_TO_ISAC],
                    "act": prev_action.copy(),
                    "grav": grav,
                }
                if hist is None:
                    hist = {k: np.tile(v, (HIST_LEN, 1)) for k, v in cur_val.items()}
                else:
                    for k in cur_val:
                        hist[k] = np.concatenate([hist[k][1:], cur_val[k][None, :]], axis=0)

                obs_enc = build_encoder_obs(motion, cur_frame, base_quat)
                tokens = stack.encode(obs_enc.reshape(1, -1))
                obs_dec = build_decoder_obs(tokens, hist)
                action = stack.decode(obs_dec.reshape(1, -1))[0].astype(np.float64)
                prev_action = action.copy()

                # action -> motor setpoint
                # remap + scale
                # not a torque hold
                q_target = DEFAULT_ANGLES_MUJOCO + action[ISAC_TO_MUJOCO] * ACTION_SCALE_MUJOCO

                # T2 replay record
                # pre-advance cursor
                dump.append({
                    "step": step,
                    "q_mj": q_mj.copy(),
                    "dq_mj": dq_mj.copy(),
                    "base_quat": base_quat.copy(),
                    "gyro": gyro.copy(),
                    "action": action.copy(),
                    "q_target": q_target.copy(),
                    "obs_enc": obs_enc.copy(),
                    "obs_dec": obs_dec.copy(),
                    "cur_frame": cur_frame,
                    "motion_n": motion.n,
                })

                z_hist.append(float(data.qpos[2]))
                x_hist.append(float(data.qpos[0]))
                t_hist.append(step * sim_dt)
                tilt_hist.append(tilt_deg(base_quat))
                n_control += 1
                cur_frame = min(cur_frame + 1, motion.n - 1)

            # ------------------------------------------------------------------
            # PD torque, 500 Hz
            #
            # deploy never sends torque
            # 50 Hz Control posts a
            # position setpoint (q, kp, kd)
            # motors do PD at ~1 kHz
            # g1_deploy_onnx_ref.cpp: rt/lowcmd
            #
            # old: PD at 50 Hz,
            # torque held 10 steps
            # zero-order hold
            # not a servo
            # q - q_target grows
            # loop blows up
            # every reference: 522 vs 139 N.m
            # sim bug, not the SONIC port
            #
            # same PD law, just at 500 Hz
            # ------------------------------------------------------------------
            q_pd = data.qpos[qadr].astype(np.float64)
            dq_pd = data.qvel[vadr].astype(np.float64)
            data.ctrl[aid] = KP_MUJOCO * (q_target - q_pd) - KD_MUJOCO * dq_pd

            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                fell = True
                break

        z = np.array(z_hist)
        x = np.array(x_hist)
        t = np.array(t_hist)
        tl = np.array(tilt_hist)
        max_tilt = float(tl.max()) if tl.size else 0.0
        z_min = float(z.min()) if z.size else float("nan")
        z_end = float(z[-1]) if z.size else float("nan")
        # mean speed, last 1 s (world x)
        if t.size >= 2:
            t_end = t[-1]
            i0 = int(np.searchsorted(t, t_end - 1.0))
            dt = t_end - t[i0]
            mean_vx = float((x[-1] - x[i0]) / dt) if dt > 1e-9 else 0.0
        else:
            mean_vx = 0.0
        # fail closed: no ticks -> nan
        # "A stands" on no evidence
        no_valid_state = bool(z.size == 0) or not (np.isfinite(z_min) and np.isfinite(max_tilt))
        fell = bool(fell or no_valid_state or max_tilt > TILT_LIMIT_DEG or z_min < Z_FALL)

        out = {
            "label": label,
            "fell": fell,
            "z_start": float(z[0]) if z.size else float("nan"),
            "z_min": z_min,
            "z_end": z_end,
            "max_tilt_deg": max_tilt,
            "mean_vx_last1s": mean_vx,
            "sim_s": float(len(z) * CONTROL_EVERY * sim_dt) if z.size else 0.0,
            "n_planner_inf": int(stack.n_planner_inf),
            "n_control_ticks": int(n_control),
        }
        print(json.dumps(out))
        return out, dump

    print(f"constants: kp[0..5]={np.round(KP_MUJOCO[:6],2).tolist()} "
          f"kd[0..5]={np.round(KD_MUJOCO[:6],2).tolist()} "
          f"action_scale[0..5]={np.round(ACTION_SCALE_MUJOCO[:6],4).tolist()} "
          f"default_angles[0..5]={DEFAULT_ANGLES_MUJOCO[:6].tolist()}")

    t0 = time.perf_counter()
    ra, da = run_test("A_tracker_only_standing", 0, 0.0, 6.0, use_planner=False)
    rb, db = run_test("B_planner_idle", 0, 0.0, 6.0, use_planner=True)
    rc, dc = run_test("C_planner_walk", 2, 1.0, 8.0, use_planner=True)
    print(f"wall_total_s={time.perf_counter() - t0:.1f}")

    if args.dump_states:
        # one npz, A/B/C
        # test_index tags each tick
        # T2 contract: don't rename
        recs = [(0, r) for r in da] + [(1, r) for r in db] + [(2, r) for r in dc]
        arr = {
            "test_index": np.array([i for i, _ in recs], dtype=np.int64),
            "step": np.array([r["step"] for _, r in recs], dtype=np.int64),
            "cur_frame": np.array([r["cur_frame"] for _, r in recs], dtype=np.int64),
            "motion_n": np.array([r["motion_n"] for _, r in recs], dtype=np.int64),
            "q_mj": np.array([r["q_mj"] for _, r in recs]),
            "dq_mj": np.array([r["dq_mj"] for _, r in recs]),
            "base_quat": np.array([r["base_quat"] for _, r in recs]),
            "gyro": np.array([r["gyro"] for _, r in recs]),
            "action": np.array([r["action"] for _, r in recs]),
            "q_target": np.array([r["q_target"] for _, r in recs]),
            "obs_enc": np.array([r["obs_enc"] for _, r in recs]),
            "obs_dec": np.array([r["obs_dec"] for _, r in recs]),
        }
        np.savez_compressed(args.dump_states, **arr)
        print(f"dumped_states: {args.dump_states} ticks={len(recs)} "
              f"(per-test: A={len(da)} B={len(db)} C={len(dc)})")

    a_stands = (not ra["fell"])
    b_stands = (not rb["fell"])
    c_stands = (not rc["fell"])
    # C must walk
    # not just not fall
    # frozen C = ignored planner
    c_advanced = abs(rc["mean_vx_last1s"]) > 0.3
    print(f"CONTROL_B_STANDS: {'true' if b_stands else 'false'}")
    print(f"CONTROL_C_WALKS: {'true' if (c_stands and c_advanced) else 'false'} "
          f"(not_fell={str(c_stands).lower()} mean_vx_last1s={rc['mean_vx_last1s']:+.3f})")
    print(f"CONTROL_A_STANDS: {'true' if a_stands else 'false'}")
    if a_stands:
        print("INTERPRETABLE: A stands, so the tracker path is exercised; "
              "B and C are interpretable as planner-driven results.")
    else:
        print("INTERPRETABLE: A does NOT stand, so the tracker path (obs layout, gains, "
              "remap) is unvalidated and B and C are MEANINGLESS.")
    # gate = printed verdict
    # rc 0 only on full pass
    # CI reads process status
    ok = a_stands and b_stands and c_stands and c_advanced
    return 0 if ok else 1
