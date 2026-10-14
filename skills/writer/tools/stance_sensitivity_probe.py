#!/usr/bin/env python3
"""Direct sensitivity probe: does the Balance ONNX's hip-roll OUTPUT depend on
the hip-roll OBSERVATION?

WHY THIS EXISTS
---------------
On the robot, `stance 4` returned DONE, the socket reported `stance=+4.0`, and
the log proves `_write_stance` rebound BOTH arrays (:922-923 run before the
`_stance_log` call that printed).  Code reading says the observation therefore
carries `-0.0698` on the hip-roll entries and the readout carries `+0.0698`.
The measured hip roll moved **0.0 deg**.

A zero that exact needs one of exactly two explanations:
  (a) the policy is DEAF to the hip-roll obs entries -> the whole stance design
      is dead by construction, regardless of the robot; or
  (b) the policy is NOT deaf, and the box's chain is broken somewhere a
      --verbose restart would reveal.

This script settles (a) vs (b) with the writer's OWN classes and the REAL ONNX,
on the laptop, touching nothing.

WHAT IT MEASURES
----------------
At a fixed operating point it reports, per commanded degree of dual shift:
  shift-obs      -- perturb obs.default_padded only (what the policy SEES)
  shift-policy   -- perturb policy.defaults only  (the readout offset)
  shift-both     -- both, exactly as _write_stance does (THE SHIPPED ONE)

and the raw partial derivative d(target_q[1]) / d(obs_frame[14]).
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WRITER = os.path.dirname(HERE)
sys.path.insert(0, WRITER)

from obs_builder import ObsBuilder          # noqa: E402
from policy_runner import PolicyRunner      # noqa: E402

CFG = os.path.join(WRITER, "config", "merge_config.yaml")
DEG = 4.0
D = DEG * np.pi / 180.0
# _write_stance touches: hip L(+), R(-),
# ankle L(+), R(-); ank = STANCE_ANKLE_RATIO*hip
ANK_RATIO = -0.85
IDX = {1: +D, 7: -D, 5: +ANK_RATIO * D, 11: -ANK_RATIO * D}


def run(pol, obs, q, dq, quat, gyro, cmd, n=12):
    """Push n ticks on the CURRENT arrays; return target_q at the end."""
    for _ in range(n):
        frame = obs.push(q, dq, quat, gyro, cmd, pol.action)
        pol.infer(frame, cmd)
    return pol.target_q.copy()


def main():
    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    onnx_dir = cfg["onnx_dir"]
    if not os.path.isabs(onnx_dir):
        onnx_dir = os.path.join(WRITER, onnx_dir)
    print(f"config      : {CFG}")
    print(f"onnx_dir    : {onnx_dir}")
    print(f"policy_bal  : {cfg.get('policy_balance')}")
    print(f"dof_pos_scale={cfg.get('dof_pos_scale')}  "
          f"action_scale={cfg.get('action_scale')}  "
          f"gate_cmd_norm={cfg.get('gate_cmd_norm')}")
    print(f"history_len = {cfg.get('obs_history_len')}  "
          f"num_actions = {cfg.get('num_actions')}")

    obs = ObsBuilder(cfg)
    pol = PolicyRunner(cfg, onnx_dir)

    # configured default pose; box drifts off it,
    # but sensitivity, not pose
    q = np.zeros(29, dtype=np.float32)
    q[:15] = np.array(cfg["default_angles"], dtype=np.float32)
    dq = np.zeros(29, dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gyro = np.zeros(3, dtype=np.float32)
    cmd = np.zeros(3, dtype=np.float32)

    base_pol = pol.defaults.copy()
    base_obs = obs.default_padded.copy()
    base = run(pol, obs, q, dq, quat, gyro, cmd)
    print(f"\nsession key : {pol.last_key}")
    print(f"BASE target_q[0:6] = {np.round(base[0:6], 4)}")
    print(f"BASE hip roll L/R  = {base[1]:+.4f} / {base[7]:+.4f} rad "
          f"({np.degrees(base[1]):+.2f} / {np.degrees(base[7]):+.2f} deg)")

    # ---- raw partial: d(out[1]) / d(obs[14]) ----
    EPS = 1e-3
    obs.default_padded = base_obs.copy()
    obs.default_padded[1] += EPS            # frame[14] = q - default -> -EPS
    obs.obs[:] = 0.0                        # wipe history, only this shows
    a1 = run(pol, obs, q, dq, quat, gyro, cmd)
    obs.default_padded = base_obs.copy()
    obs.obs[:] = 0.0
    a0 = run(pol, obs, q, dq, quat, gyro, cmd)
    d_out = (a1[1] - a0[1]) / (-EPS)        # per frame[14] unit
    print(f"\nRAW PARTIAL  d(target_q[1]) / d(frame[14]) = {d_out:+.4f}")
    print("             (sim reference: ~+0.55 gross of the +d readout term)")

    modes = ("shift-obs", "shift-policy", "shift-both")
    rows = []
    for mode in modes:
        pol.defaults = base_pol.copy()
        obs.default_padded = base_obs.copy()
        obs.obs[:] = 0.0
        if mode in ("shift-obs", "shift-both"):
            for i, dv in IDX.items():
                obs.default_padded[i] += dv
        if mode in ("shift-policy", "shift-both"):
            for i, dv in IDX.items():
                pol.defaults[i] += dv
        t = run(pol, obs, q, dq, quat, gyro, cmd)
        rows.append((mode, t))

    pol.defaults = base_pol.copy()
    obs.default_padded = base_obs.copy()
    obs.obs[:] = 0.0
    ctrl = run(pol, obs, q, dq, quat, gyro, cmd)

    print(f"\ncommanded {DEG:+.1f} deg dual shift "
          f"(hip L +{DEG:.1f}, hip R -{DEG:.1f}, "
          f"ankle L {ANK_RATIO * DEG:+.2f}, ankle R {-ANK_RATIO * DEG:+.2f})"
          f"\n  hip roll is indices 1 (L) and 7 (R); ankle roll 5 and 11."
          f"\n\n  {'mode':<14} {'d hipL':>9} {'d hipR':>9} {'d SEP-ish':>10} "
          f"{'d ankL':>9} {'d ankR':>9}")
    for mode, t in rows:
        dhL = np.degrees(t[1] - ctrl[1])
        dhR = np.degrees(t[7] - ctrl[7])
        daL = np.degrees(t[5] - ctrl[5])
        daR = np.degrees(t[11] - ctrl[11])
        print(f"  {mode:<14} {dhL:+8.3f}deg {dhR:+8.3f}deg "
              f"{abs(dhL) + abs(dhR):+9.3f} {daL:+8.3f}deg {daR:+8.3f}deg")

    print("\nINTERPRETATION")
    print("  shift-both hip-roll delta ~0 at the LIBRARY level  ->")
    print("      the mechanism is dead by construction; the sim probe was")
    print("      not measuring this code path.  Do NOT restart the writer.")
    print("  shift-both hip-roll delta clearly non-zero         ->")
    print("      the mechanism is live in the real classes; the robot-side")
    print("      break needs the --verbose leg_target read.")


if __name__ == "__main__":
    main()
