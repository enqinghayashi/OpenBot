#!/usr/bin/env python3
"""obs_unit_test.py — laptop gate for the C3 obs stack (no robot, no SDK).

Validates the 516-D obs layout, history ordering, gate, and ONNX round-trip
against the verified GR00T constants. Exit 0 = green.

Usage:  python3 tools/obs_unit_test.py [--onnx-dir <dir>]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from obs_builder import ObsBuilder, FRAME_DIM, NUM_ACTIONS
from policy_runner import PolicyRunner

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CFG = os.path.join(PKG, "config", "merge_config.yaml")

FAILS = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=os.path.join(PKG, "policy"))
    args = ap.parse_args()

    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    obs = ObsBuilder(cfg)
    defaults = np.array(cfg["default_angles"], dtype=np.float32)
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)

    # --- fake lowstate ---
    q35 = np.zeros(35, dtype=np.float32)
    q35[:15] = defaults + np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03,
                                    0.01, 0.02, -0.01, -0.02, 0.03, 0.01, 0.0, 0.0, 0.0])
    q35[15:29] = np.linspace(-0.5, 0.5, 14)          # arm joints
    dq35 = np.linspace(-0.5, 0.5, 35)
    quat_id = np.array([1.0, 0.0, 0.0, 0.0])
    gyro = np.array([0.1, -0.2, 0.3])
    prev_act = np.full(NUM_ACTIONS, 0.25)

    # --- frame map ---
    f = obs.build_frame(q35, dq35, quat_id, gyro, (0.0, 0.0, 0.0), prev_act)
    check("cmd slice [0:3] = cmd x cmd_scale", np.allclose(f[0:3], 0.0), str(f[0:3]))
    check("height [3] = 0.74 raw", abs(f[3] - 0.74) < 1e-6, str(f[3]))
    check("rpy [4:7] = rpy_cmd", np.allclose(f[4:7], cfg["rpy_cmd"]), str(f[4:7]))
    check("angvel [7:10] x 0.5", np.allclose(f[7:10], gyro * 0.5), str(f[7:10]))
    check("gravity [10:13] ~ [0,0,-1]", np.allclose(f[10:13], [0.0, 0.0, -1.0], atol=1e-6),
          str(f[10:13]))
    expect_pos = (q35[:29] - obs.default_padded) * 1.0
    check("q [13:42] = (q - default) x 1.0", np.allclose(f[13:42], expect_pos, atol=1e-6),
          f"maxdiff {np.max(np.abs(f[13:42] - expect_pos)):.2e}")
    expect_vel = dq35[:29] * 0.05
    check("dq [42:71] x 0.05", np.allclose(f[42:71], expect_vel, atol=1e-6),
          f"maxdiff {np.max(np.abs(f[42:71] - expect_vel)):.2e}")
    check("prev action [71:86]", np.allclose(f[71:86], prev_act), str(f[71:86]))
    check("frame dim = 86", f.shape == (FRAME_DIM,), str(f.shape))

    # --- history: oldest-first, zeros till 6 pushes ---
    obs.reset()
    obs.push(q35, dq35, quat_id, gyro, (0.0, 0.0, 0.0), np.zeros(NUM_ACTIONS))
    after1 = obs.obs.copy()
    obs.push(q35 + 0.01, dq35, quat_id, gyro, (0.0, 0.0, 0.0),
             np.full(NUM_ACTIONS, 0.5))
    check("obs length 516", obs.obs.shape == (516,), str(obs.obs.shape))
    # 2 pushes: [0:344] zeros, frame1 [344:430], frame2 [430:516]
    # newest at the END (oldest-first)
    check("history zero-prefix until full",
          np.all(np.abs(obs.obs[0:344]) < 1e-6))
    check("frame1 at slot 4, frame2 at slot 5 (newest end)",
          np.allclose(obs.obs[344:430], after1[430:516]) and
          np.any(np.abs(obs.obs[430:516] - obs.obs[344:430]) > 1e-6),
          "oldest-first, newest at the end")

    # --- quat_rotate_inverse sanity ---
    g = obs.quat_rotate_inverse([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    check("quat identity -> [0,0,-1]", np.allclose(g, [0.0, 0.0, -1.0], atol=1e-6), str(g))
    # body +90 deg about x: y-axis along world -z,
    # so world (0,0,-1) -> (0,-1,0) in body
    c, s = 0.70710678, 0.70710678
    g = obs.quat_rotate_inverse([c, s, 0.0, 0.0], [0.0, 0.0, -1.0])
    check("quat 90deg-x -> [0,-1,0]", np.allclose(g, [0.0, -1.0, 0.0], atol=1e-6), str(g))

    # --- the gate ---
    pol = PolicyRunner(cfg, args.onnx_dir)
    check("gate zero -> balance", pol.gate([0.0, 0.0, 0.0]) == "balance")
    check("gate small -> balance", pol.gate([0.01, 0.0, 0.0]) == "balance")
    check("gate 0.3 vx -> walk", pol.gate([0.3, 0.0, 0.0]) == "walk")
    check("gate 0.3 vy -> walk", pol.gate([0.0, 0.3, 0.0]) == "walk")

    # --- ONNX pass ---
    obs.reset()
    obs.push(q35, dq35, quat_id, gyro, (0.3, 0.0, 0.0), pol.action)
    target = pol.infer(obs.obs, (0.3, 0.0, 0.0))
    check("infer output (15,)", target.shape == (15,), str(target.shape))
    check("infer finite", np.all(np.isfinite(target)), str(target))
    check("targets within joint range",
          np.all(np.abs(target) < 3.0),
          f"min {target.min():.2f} max {target.max():.2f}")
    check("target = act*0.25 + defaults",
          np.allclose(target, pol.action * 0.25 + defaults, atol=1e-5))
    check("walk key after 0.3 cmd", pol.last_key == "walk")

    # no cmd -> balance
    obs.reset()
    obs.push(q35, dq35, quat_id, gyro, (0.0, 0.0, 0.0), pol.action)
    target0 = pol.infer(obs.obs, (0.0, 0.0, 0.0))
    check("balance key after zero cmd", pol.last_key == "balance")
    check("balance targets near defaults",
          np.max(np.abs(target0 - defaults)) < 0.5,
          f"maxdiff {np.max(np.abs(target0 - defaults)):.3f}")

    print()
    if FAILS:
        print(f"obs_unit_test FAILED: {len(FAILS)} -> {FAILS}")
        sys.exit(1)
    print("obs_unit_test GREEN (all checks passed)")
    sys.exit(0)
