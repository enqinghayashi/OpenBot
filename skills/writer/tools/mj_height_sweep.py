#!/usr/bin/env python3
"""mj_height_sweep.py — laptop-only MuJoCo height-reference sweep (no robot).

Drives the REAL GR00T Balance policy through the EXACT obs stack the writer
uses (obs_builder.ObsBuilder + policy_runner.PolicyRunner) in g1_gear_wbc.xml,
ramping the policy HEIGHT REFERENCE f[3] (obs.height_cmd) DOWN, measuring the
settled response at each height, and rendering video so the operator can SEE
the squat.

Why mutating obs.height_cmd reaches the actor (CONFIRMED-TRACE):
  ObsBuilder.__init__ stores height_cmd once, but build_frame() dereferences
  self.height_cmd LIVE on every call (obs_builder.py:101, `f[3] = self.height_cmd`)
  and push() copies that freshly-built frame into the obs tail
  (obs_builder.py:111). So assigning `obs.height_cmd = <new>` before the next
  obs.push() reaches the actor on the VERY NEXT policy tick (~20 ms at 50 Hz).
  Caveat: only the newest of the 6 history frames carries the new value; the
  other 5 age out over 6 ticks (~0.12 s) — a smooth ramp hides that latency.

Modes:
  stepped = reset to --start-height, settle, command each rung for --hold-s
            (instant step), measure settled.
  ramped  = continuous --ramp-s descent (the rendered one).  JUDGES ONLY the
            endpoint rung's settled tail; intermediate rungs are crossed, not
            judged — the summary now says so explicitly.
  inband  = settle, then a SLEW-LIMITED (writer height_slew_rate) settled
            stepped sweep over the WRITER'S ACTUAL DEPLOYED BAND
            [height_cmd_min, height_cmd_max] read from the config, with each
            rung clamped to that band.  This is the transient the deployed
            writer really produces (the writer rate-limits every command and
            clamps it into the band); every rung is settled and judged.
  recover = the ASCENT question.  Per ladder rung: RESET at --start-height
            (standing), slew the reference DOWN to the rung at the writer slew
            rate, HOLD until settled, then slew it straight BACK UP to standing
            at the same rate and measure whether the policy stands back up.
            Per rung: RECOVERED (settled back within --hold-tol of standing),
            final settled z, time to ascend (s), overshoot, and whether it fell.
            Descending is gravity-assisted; this measures the concentric push
            out of the deep lean with the heels lifted.
  chained = ONE continuous run, NO re-settling between rungs: a single
            writer-rate slew DOWN through the whole ladder to the deepest rung,
            then straight back UP to standing.  The realistic single-command
            usage pattern (not a staircase of independent re-settles).  Per-rung
            crossings are reported on BOTH legs (momentary, not settled) plus
            the final standing state.

Fall detection is HEIGHT-AWARE: mj_policy_check.py:141's hardcoded
`qpos[2] < 0.4` scores a legal deep squat as a FALL and is NOT reused; here
fall <=> z < height_cmd - fall_drop, or |roll|/|pitch| > max_tilt, or
z < min_abs_z.  The gate is NEVER weakened to make a rung pass.

ASCENT-AWARE GATE (recover / chained only).  `z < height_cmd - fall_drop` is a
LOWER-BOUND test on the reference.  It is sound while height_cmd is
NON-INCREASING (descent / steady): lag then leaves z ABOVE the reference, and
losing > fall_drop below a settled command is a real collapse.  While the
reference RISES the roles invert — the reference leads the body, tracking lag
pulls z BELOW height_cmd, and the raw test fires on ordinary lag, not a fall
(a STEPPED rise, a fast --slew-rate, or a recovery from a perturbed pose all
false-trip).  So recover/chained gate on the RUNNING MINIMUM of height_cmd over
the segment, `gate_ref = min_{s<=t} height_cmd(s)`, with the SAME fall_drop:
    fall <=> z < gate_ref - fall_drop  OR  |roll| > max_tilt
             OR  |pitch| > max_tilt  OR  z < min_abs_z
PROOF the descent gate is unchanged: on any non-increasing segment
min_{s<=t} height_cmd(s) == height_cmd(t), so gate_ref == height_cmd and the
test is bit-identical to the plain descent gate (this is exactly why a running
MINIMUM is used and not a running PEAK — a peak-relative gate would false-trip
every legal deep squat on the way down, weakening the descent gate, which is
forbidden).  On the rising leg gate_ref stays frozen at the deepest commanded
height, i.e. the gate becomes "do not collapse below the deepest rung you were
actually told to hold" — the lower bound that stays meaningful while the body
pushes up.  The tilt and min_abs_z terms are ascent-safe and UNCHANGED.  In
every other mode ascent_aware is False, so those modes keep the literal
`z < height_cmd - fall_drop` gate unchanged.

Usage: python3 tools/mj_height_sweep.py --mode ramped  |  --mode stepped
       python3 tools/mj_height_sweep.py --mode inband
       python3 tools/mj_height_sweep.py --mode recover   (ascent / recovery)
       python3 tools/mj_height_sweep.py --mode chained   (ascent / recovery)
"""

import argparse
import json
import os
import sys
import time

# OSMesa isn't on this box; EGL works here
# Renderer dies with
# "'NoneType' object has no attribute 'glGetError'"
# (probe: 320x240 uint8 OK)
# setdefault, so an override wins
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import mujoco

from obs_builder import ObsBuilder
from policy_runner import PolicyRunner

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CFG = os.path.join(PKG, "config", "merge_config.yaml")
XML_DEFAULT = os.path.join(PKG, "sim", "g1_gear_wbc.xml")

# XML joint names
# doc order == G1JointIndex 0..14
POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

# report indices (== G1JointIndex)
I_LKNEE, I_LANKP = 3, 4
I_RKNEE, I_RANKP = 9, 10

DEFAULT_LADDER = [0.74, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]


def band_rungs(lo, hi, stand, step=0.05):
    """Descending rungs spanning the writer's deployed band [lo, hi].

    Steps down from hi by `step`, always includes lo and the nominal standing
    height when it lies inside the band.  Every returned rung is inside
    [lo, hi] — the same interval obs_builder.set_height / the writer clamps to,
    so no rung here can exercise a height the deployed writer could not.
    """
    rungs = []
    h = hi
    while h > lo + 1e-9:
        rungs.append(round(h, 3))
        h -= step
    rungs.append(round(lo, 3))
    if lo - 1e-9 <= stand <= hi + 1e-9:
        rungs.append(round(stand, 3))
    return sorted({r for r in rungs}, reverse=True)


def _win_median(ts, zs, t_a, t_b):
    """Median z over [t_a, t_b]; NaN if the window is empty (a truncated trace)."""
    ts = np.asarray(ts)
    zs = np.asarray(zs)
    if ts.size == 0:
        return float("nan")
    sel = (ts >= t_a) & (ts <= t_b)
    return float(np.median(zs[sel])) if bool(sel.any()) else float("nan")


def _win_max(ts, zs, t_a, t_b):
    """Peak z over [t_a, t_b]; NaN if the window is empty."""
    ts = np.asarray(ts)
    zs = np.asarray(zs)
    if ts.size == 0:
        return float("nan")
    sel = (ts >= t_a) & (ts <= t_b)
    return float(zs[sel].max()) if bool(sel.any()) else float("nan")


def _first_at_or_above(ts, zs, thr, t_from=0.0):
    """First t >= t_from with z >= thr; NaN if never reached (truncated trace)."""
    ts = np.asarray(ts)
    zs = np.asarray(zs)
    if ts.size == 0:
        return float("nan")
    sel = np.flatnonzero((ts >= t_from) & (zs >= thr))
    return float(ts[int(sel[0])]) if sel.size else float("nan")


def quat_roll_pitch(q):
    """roll, pitch (rad) from quat wxyz — magnitude only, for the tilt gate."""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-9:
        return 0.0, 0.0
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.environ.get("MERGE_MUJOCO_XML", XML_DEFAULT))
    ap.add_argument("--onnx-dir", default=os.path.join(PKG, "policy"))
    ap.add_argument("--mode",
                    choices=["stepped", "ramped", "inband", "recover", "chained"],
                    default="ramped",
                    help="inband = slew-limited settled stepped sweep across "
                         "the writer's real deployed band; recover = per-rung "
                         "slew down then back up to standing (ascent); "
                         "chained = one continuous down-through-ladder then up "
                         "(ascent).  See docstring")
    ap.add_argument("--ladder", type=float, nargs="+", default=None,
                    help="height rungs, m (descending).  Default: DEFAULT_LADDER "
                         "for stepped/ramped; the writer's deployed band for "
                         "inband; the band's squat depths (rungs <= standing) "
                         "for recover/chained")
    ap.add_argument("--slew-rate", type=float, default=None,
                    help="command rate limit, m/s (inband/recover/chained).  "
                         "Default: the writer's height_slew_rate from the config "
                         "— the rate the deployed writer actually produces")
    ap.add_argument("--start-height", type=float, default=0.74,
                    help="reset base z AND initial height_cmd (config constant "
                         "0.74 is a default, not a hardcoded reset)")
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="settle at --start-height before the rung is commanded")
    ap.add_argument("--hold-s", type=float, default=6.0,
                    help="seconds commanded at each rung (stepped; also the "
                         "squat hold and the standing hold in recover)")
    ap.add_argument("--ramp-s", type=float, default=8.0,
                    help="seconds to ramp start -> deepest rung (ramped)")
    ap.add_argument("--measure-s", type=float, default=2.0,
                    help="window (s) at the end of a segment used as 'settled'")
    # fall gate, all CLI-tunable
    # fall_drop 0.12 m
    # a healthy tracker holds a few cm
    # >12 cm below command is a collapse
    # (loose enough that a transient
    #  undershoot won't trip)
    # max_tilt 0.6 rad (~34 deg)
    # past any legal squat lean,
    # still inside a real topple
    # min_abs_z 0.15 m = floor contact
    # (pelvis mesh ~0.06 m)
    ap.add_argument("--fall-drop", type=float, default=0.12)
    ap.add_argument("--max-tilt", type=float, default=0.6)
    ap.add_argument("--min-abs-z", type=float, default=0.15)
    ap.add_argument("--hold-tol", type=float, default=0.08,
                    help="|settled z - commanded h| <= tol counts as HELD")
    ap.add_argument("--asym-warn", type=float, default=0.15,
                    help="|knee_L - knee_R| above this is called out loudly")
    ap.add_argument("--render", choices=["auto", "on", "off"], default="auto",
                    help="auto = ramped only (recover/chained video needs --render "
                         "on); on = always; off = physics only")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--cam-distance", type=float, default=2.4)
    ap.add_argument("--cam-azimuth", type=float, default=90.0, help="90 = side view")
    ap.add_argument("--cam-elevation", type=float, default=-10.0)
    ap.add_argument("--cam-height", type=float, default=0.45,
                    help="FIXED lookat z so the descent is visible against the "
                         "floor (tracking z would pin the robot on screen)")
    ap.add_argument("--out", default=os.path.join(HERE, "out_height_sweep"))
    ap.add_argument("--omega-mode", choices=["reference", "body"], default="reference",
                    help="obs[7:10] source, same convention as mj_policy_check")
    args = ap.parse_args()
    if args.hold_s <= args.measure_s:
        ap.error("--hold-s must exceed --measure-s")
    os.makedirs(args.out, exist_ok=True)
    with open(CFG) as f:
        cfg = yaml.safe_load(f)

    # --- the band the writer REALLY runs (clamp) and its slew (rate limit) ---
    # writer clamps into
    # [height_cmd_min, height_cmd_max]
    # and rate-limits at height_slew_rate
    # (same rate limit as obs_builder.set_height)
    # inband copies both, so the
    # transient is the real one
    # older configs: height_cmd
    height_stand = float(cfg.get("height_cmd", 0.74))
    band_lo = float(cfg.get("height_cmd_min", height_stand))
    band_hi = float(cfg.get("height_cmd_max", height_stand))
    if band_lo > band_hi:
        band_lo, band_hi = band_hi, band_lo
    slew = (float(args.slew_rate) if args.slew_rate is not None
            else float(cfg.get("height_slew_rate", 0.15)))

    if args.ladder is not None:
        ladder = [float(h) for h in args.ladder]
    elif args.mode == "inband":
        ladder = band_rungs(band_lo, band_hi, height_stand)
    elif args.mode in ("recover", "chained"):
        # squat depths only
        # a rung above standing isn't a squat
        # (recovery targets standing) — sweep
        # [height_cmd_min, min(height_cmd_max, standing)]
        full = band_rungs(band_lo, band_hi, height_stand)
        ladder = [h for h in full if h <= args.start_height + 1e-9]
        skipped = [h for h in full if h > args.start_height + 1e-9]
        if skipped:
            print(f"{args.mode}: rungs above standing "
                  f"({', '.join(f'{h:.3f}' for h in skipped)}) are not squat "
                  f"depths — excluded")
    else:
        ladder = list(DEFAULT_LADDER)
    if args.mode == "inband":
        assert min(ladder) >= band_lo - 1e-9 and max(ladder) <= band_hi + 1e-9, \
            f"inband rungs must lie in [{band_lo}, {band_hi}]"
        print(f"inband: writer band [{band_lo:.3f}, {band_hi:.3f}] m "
              f"(height_cmd_min/max), slew {slew:.3f} m/s "
              f"(height_slew_rate), stand {height_stand:.3f} m")
        print(f"inband rungs ({len(ladder)}): "
              + ", ".join(f"{h:.3f}" for h in ladder))
    if args.mode in ("recover", "chained"):
        if not ladder:
            ap.error(f"{args.mode}: no ladder rungs <= --start-height "
                     f"({args.start_height}) inside the deployed band "
                     f"[{band_lo:.3f}, {band_hi:.3f}]")
        print(f"{args.mode}: standing {args.start_height:.3f} m, band "
              f"[{band_lo:.3f}, {band_hi:.3f}] m, slew {slew:.3f} m/s "
              f"(height_slew_rate)")
        print(f"{args.mode} squat rungs ({len(ladder)}): "
              + ", ".join(f"{h:.3f}" for h in ladder))

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    sim_dt = model.opt.timestep
    policy_every = max(1, int(round(0.02 / sim_dt)))   # 50 Hz tick
    print(f"sim_dt={sim_dt} policy_every={policy_every} "
          f"policy_rate={1.0/(policy_every*sim_dt):.0f} Hz")

    # policy joint -> qpos/qvel/actuator index
    # by NAME
    # (mj_policy_check.py: by name,
    #  not by position)
    jnt_id = {model.joint(i).name: i for i in range(model.njnt)}
    act_id = {model.actuator(i).name: i for i in range(model.nu)}
    qid = np.zeros(15, dtype=int)
    vid = np.zeros(15, dtype=int)
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

    # arm/hand actuators:
    # PDs them to 0
    # (kp 100, kd 0.5)
    arm_act = [i for i in range(model.nu) if i not in aid.tolist()]
    arm_kp = np.full(len(arm_act), 100.0)
    arm_kd = np.full(len(arm_act), 0.5)

    # --- offscreen renderer (fail gracefully, never take down the physics) ---
    want_render = args.render == "on" or (args.render == "auto" and args.mode == "ramped")
    renderer = openbot = None
    render_ok = False
    if want_render:
        try:
            renderer = mujoco.Renderer(model, height=args.height, width=args.width)
            openbot = mujoco.MjvCamera()   # free camera
            openbot.azimuth, openbot.elevation = args.cam_azimuth, args.cam_elevation
            openbot.distance, openbot.lookat[:] = args.cam_distance, [0.0, 0.0, args.cam_height]
            render_ok = True
            print(f"render: {args.width}x{args.height} offscreen OK "
                  f"(MUJOCO_GL={os.environ.get('MUJOCO_GL')})")
        except Exception as exc:  # noqa: BLE001
            renderer = None
            print(f"render: UNAVAILABLE ({type(exc).__name__}: {exc}) — physics "
                  f"still runs; qpos trajectory saved for later rendering")

    def render_frame():
        # follow base x/y
        # lookat z FIXED
        # (tracking z would pin the robot on screen)
        openbot.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), args.cam_height]
        renderer.update_scene(data, camera=openbot)
        return renderer.render().copy()

    def run(cmd, height_fn, seconds, label, h_target, capture=False,
            ascent_aware=False):
        """Roll the policy for `seconds`. height_fn(t) -> commanded height each
        tick; `h_target` is the settled height this segment is judged against.

        ascent_aware=False (every non-ascent mode): the fall gate is the plain
        `z < height_cmd - fall_drop` — byte-identical to the historical gate.
        ascent_aware=True (recover/chained): the gate reference is the RUNNING
        MINIMUM of height_cmd over the segment, so a RISING reference cannot
        false-trip on tracking lag.  On any non-increasing segment the running
        minimum EQUALS height_cmd, so the gate is bit-identical to the descent
        gate there — descent behaviour is provably unchanged (module docstring).
        """
        mujoco.mj_resetData(model, data)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])   # identity wxyz quat
        data.qpos[qid] = defaults                          # all 15 policy joints
        data.qpos[2] = args.start_height                   # not cfg['height_cmd']
        mujoco.mj_forward(model, data)
        obs.reset()
        obs.height_cmd = args.start_height                 # seed with the reference
        n_steps = int(seconds / sim_dt)
        render_stride = max(1, int(round(1.0 / (args.fps * sim_dt))))
        prev_act = np.zeros(15, dtype=np.float32)
        target = defaults.copy()
        frames, qpos_track = [], []
        z, kl, kr, alp, arp, pit, rol, xs, ys, ts, hs = ([] for _ in range(11))
        fall, reason = False, ""
        gate_ref = None                       # running min, ascent only
        t0 = time.perf_counter()
        for step in range(n_steps):
            t = step * sim_dt
            hc = float(height_fn(t))
            if ascent_aware:                  # O(1) scalar, no alloc
                gate_ref = hc if gate_ref is None or hc < gate_ref else gate_ref
            if step % policy_every == 0:
                obs.height_cmd = hc      # the sweep seam
                q29 = data.qpos[7:36].astype(np.float32)
                dq29 = data.qvel[6:35].astype(np.float32)
                quat = data.qpos[3:7].astype(np.float32)      # wxyz order
                if args.omega_mode == "reference":
                    gyro = data.qvel[3:6].astype(np.float32)  # NVIDIA convention
                else:
                    gyro = obs.quat_rotate_inverse(quat, data.qvel[0:3])
                obs_516 = obs.push(q29, dq29, quat, gyro, cmd, prev_act)
                target = pol.infer(obs_516, cmd)
                prev_act = pol.action.copy()
            # PD torque
            # (no firmware controller in sim)
            q = data.qpos[qid]
            dq = data.qvel[vid]
            data.ctrl[aid] = kp * (target - q) - kd * dq
            q_arm = data.qpos[7:36][15:]
            dq_arm = data.qvel[6:35][15:]
            data.ctrl[arm_act] = arm_kp * (0.0 - q_arm) - arm_kd * dq_arm
            mujoco.mj_step(model, data)
            if render_ok and step % render_stride == 0:
                frames.append(render_frame())
            if capture:
                qpos_track.append(data.qpos.copy())
            zb = float(data.qpos[2])
            roll, pitch = quat_roll_pitch(data.qpos[3:7])
            jq = data.qpos[qid]                    # the 15 joints
            z.append(zb); ts.append(t); hs.append(hc)
            xs.append(float(data.qpos[0])); ys.append(float(data.qpos[1]))
            kl.append(float(jq[I_LKNEE])); kr.append(float(jq[I_RKNEE]))
            alp.append(float(jq[I_LANKP])); arp.append(float(jq[I_RANKP]))
            pit.append(pitch); rol.append(roll)
            # ascent-aware gate ref
            # None in non-ascent modes
            # -> drop_ref is hc
            # -> old descent gate, exactly
            # while rising it pins at
            # the deepest command
            # tilt/min_abs_z ascent-safe
            drop_ref = hc if gate_ref is None else gate_ref
            if zb < drop_ref - args.fall_drop:
                fall = True
                reason = (f"z<cmd-{args.fall_drop:.2f}" if gate_ref is None
                          else f"z<mincmd-{args.fall_drop:.2f}")
            elif abs(roll) > args.max_tilt:
                fall, reason = True, f"|roll|={abs(roll):.2f}"
            elif abs(pitch) > args.max_tilt:
                fall, reason = True, f"|pitch|={abs(pitch):.2f}"
            elif zb < args.min_abs_z:
                fall, reason = True, f"z<{args.min_abs_z:.2f}"
            if fall:
                print(f"[{label}] FALL at t={t:.2f}s z={zb:.3f} "
                      f"cmd={hc:.2f} ({reason})")
                break
        wall = time.perf_counter() - t0
        m = int(max(1, round(args.measure_s / sim_dt)))
        sl = slice(-m, None)
        zz = np.asarray(z)
        res = {
            "label": label,
            "h_cmd": float(h_target),
            "z_settle": float(np.median(zz[sl])),
            "z_p2p": float(zz[sl].max() - zz[sl].min()),
            "knee_L": float(np.mean(np.asarray(kl)[sl])),
            "knee_R": float(np.mean(np.asarray(kr)[sl])),
            "ank_pitch_L": float(np.mean(np.asarray(alp)[sl])),
            "ank_pitch_R": float(np.mean(np.asarray(arp)[sl])),
            "pitch": float(np.mean(np.asarray(pit)[sl])),
            "roll": float(np.mean(np.asarray(rol)[sl])),
            "x_drift": float(np.asarray(xs)[sl][-1] - np.asarray(xs)[sl][0]),
            "y_drift": float(np.asarray(ys)[sl][-1] - np.asarray(ys)[sl][0]),
            "fall": fall, "fall_reason": reason, "wall": wall, "sim": seconds,
        }
        res["knee_asym"] = abs(res["knee_L"] - res["knee_R"])
        res["held"] = (not fall) and abs(res["z_settle"] - h_target) <= args.hold_tol
        trace = {
            "t": np.asarray(ts), "z": zz, "h": np.asarray(hs),
            "knee_L": np.asarray(kl), "knee_R": np.asarray(kr),
            "qpos": np.asarray(qpos_track) if qpos_track else None,
            "frames": frames,
        }
        return res, trace
    HEAD = (f"{'cmd_h':>6} {'z_set':>6} {'z_p2p':>6} {'kneL':>7} {'kneR':>7} "
            f"{'dKnee':>6} {'ankL':>7} {'ankR':>7} {'pitch':>6} {'roll':>6} "
            f"{'dx':>6} {'dy':>6} {'held':>5}")
    print(HEAD)
    print("-" * len(HEAD))

    def row(r):
        return (f"{r['h_cmd']:6.3f} {r['z_settle']:6.3f} {r['z_p2p']:6.3f} "
                f"{r['knee_L']:7.3f} {r['knee_R']:7.3f} {r['knee_asym']:6.3f} "
                f"{r['ank_pitch_L']:7.3f} {r['ank_pitch_R']:7.3f} "
                f"{r['pitch']:6.3f} {r['roll']:6.3f} "
                f"{r['x_drift']:6.3f} {r['y_drift']:6.3f} "
                f"{'HELD' if r['held'] else 'FAIL':>5}")
    def warn_asym(res, where):
        if res["knee_asym"] > args.asym_warn:
            print(f"  !! L/R KNEE ASYMMETRY {res['knee_asym']:.3f} rad "
                  f"(> {args.asym_warn:.2f}) at {where} — KNOWN "
                  f"lateral-sign-mirror hazard, treat as a red flag")

    results, traces, ok = [], {}, True
    primary_label = None
    if args.mode == "stepped":
        for rung in ladder:
            def hfn(t, rung=rung):
                return args.start_height if t < args.settle_s else rung
            label = f"h={rung:.2f}"
            res, tr = run((0.0, 0.0, 0.0), hfn,
                          args.settle_s + args.hold_s, label, rung)
            results.append(res)
            traces[label] = tr
            print(row(res))
            warn_asym(res, label)
            ok &= res["held"]
    elif args.mode == "inband":
        # the transient the writer makes
        # settle, slew to the rung
        # at height_slew_rate, hold
        # every rung settled and judged
        # inside the band
        for rung in ladder:
            span = rung - args.start_height
            travel = 0.0 if abs(span) < 1e-9 else abs(span) / max(slew, 1e-9)
            sign = 1.0 if span >= 0.0 else -1.0

            def hfn(t, rung=rung, span=span, travel=travel, sign=sign):
                if t < args.settle_s:
                    return args.start_height
                dt = t - args.settle_s
                if abs(span) < 1e-9 or dt >= travel:
                    return rung
                # the writer's rate limit
                return args.start_height + sign * slew * dt

            label = f"h={rung:.3f}"
            dur = args.settle_s + travel + args.hold_s
            res, tr = run((0.0, 0.0, 0.0), hfn, dur, label, rung)
            res["slew_s"] = travel
            results.append(res)
            traces[label] = tr
            print(row(res))
            warn_asym(res, label)
            ok &= res["held"]
    elif args.mode == "recover":
        # the ASCENT question
        # per rung: reset at standing
        # slew DOWN at the writer rate
        # HOLD until settled
        # slew BACK UP to standing
        # gate is ascent-aware (running min)
        # so the rising leg can't
        # false-trip on tracking lag
        HEAD_R = (f"{'rung':>6} {'z_rung':>7} {'z_final':>7} {'t_asc':>6} "
                  f"{'overs':>6} {'reached':>7} {'fell':>5} {'RECOV':>6}")
        print(HEAD_R)
        print("-" * len(HEAD_R))
        primary_label = f"h={min(ladder):.3f}"     # deepest rung -> .npz
        for rung in ladder:
            span = rung - args.start_height
            sign = 1.0 if span >= 0.0 else -1.0
            t_leg = 0.0 if abs(span) < 1e-9 else abs(span) / max(slew, 1e-9)
            t_dn_end = args.settle_s + t_leg
            t_up_beg = t_dn_end + args.hold_s
            t_up_end = t_up_beg + t_leg

            def hfn(t, rung=rung, sign=sign, t_leg=t_leg, t_dn_end=t_dn_end,
                    t_up_beg=t_up_beg, t_up_end=t_up_end):
                if t < args.settle_s:
                    return args.start_height                 # settle at standing
                if abs(rung - args.start_height) < 1e-9:
                    return rung
                if t < t_dn_end:                             # slew DOWN to the rung
                    return args.start_height + sign * slew * (t - args.settle_s)
                if t < t_up_beg:                             # hang out at the rung
                    return rung
                if t < t_up_end:                             # slew back UP to standing
                    return rung - sign * slew * (t - t_up_beg)
                return args.start_height                     # what we measure

            label = f"h={rung:.3f}"
            dur = t_up_end + args.hold_s
            res, tr = run((0.0, 0.0, 0.0), hfn, dur, label, rung,
                          capture=True, ascent_aware=True)
            tsv, zsv = tr["t"], tr["z"]
            # squat depth it settled at
            # (last measure window of the hold)
            res["z_rung_settle"] = _win_median(tsv, zsv,
                                               t_up_beg - args.measure_s, t_up_beg)
            # recovery: first z climb
            # back within --hold-tol of standing
            t_hit = _first_at_or_above(tsv, zsv,
                                       args.start_height - args.hold_tol,
                                       t_from=t_up_beg)
            res["t_ascend_s"] = ((t_hit - t_up_beg) if np.isfinite(t_hit)
                                 else float("nan"))
            pk = _win_max(tsv, zsv, t_up_beg, np.inf)    # peak on the rise
            res["overshoot"] = (max(0.0, pk - args.start_height)
                                if np.isfinite(pk) else float("nan"))
            res["stand_height"] = args.start_height
            res["recovered"] = ((not res["fall"])
                                and abs(res["z_settle"] - args.start_height)
                                <= args.hold_tol)
            res["rung_reached"] = ((not res["fall"])
                                   and np.isfinite(res["z_rung_settle"])
                                   and abs(res["z_rung_settle"] - rung)
                                   <= args.hold_tol)
            res["held"] = res["recovered"] and res["rung_reached"]
            results.append(res)
            traces[label] = tr
            reached_s = "yes" if res["rung_reached"] else "NO"
            fell_s = "yes" if res["fall"] else "no"
            recov_s = "YES" if res["recovered"] else "NO"
            print(f"{rung:6.3f} {res['z_rung_settle']:7.3f} "
                  f"{res['z_settle']:7.3f} {res['t_ascend_s']:6.2f} "
                  f"{res['overshoot']:6.3f} {reached_s:>7} {fell_s:>5} "
                  f"{recov_s:>6}")
            warn_asym(res, label)
            ok &= res["held"]
    elif args.mode == "chained":
        # ONE continuous run
        # no re-settling: the realistic
        # single-command pattern
        # one writer-rate slew DOWN to the
        # deepest rung, then straight back UP
        # then a hold to measure
        # per-rung crossings are momentary
        # only the final standing state
        # gets JUDGED
        deep = min(ladder)
        span = deep - args.start_height
        t_leg = 0.0 if abs(span) < 1e-9 else abs(span) / max(slew, 1e-9)
        t_bot = args.settle_s + t_leg
        t_top = t_bot + t_leg
        total = t_top + args.hold_s

        def hfn(t, deep=deep, span=span, t_leg=t_leg, t_bot=t_bot, t_top=t_top):
            if t < args.settle_s:
                return args.start_height
            if abs(span) < 1e-9:
                return deep
            if t < t_bot:                                    # still going DOWN
                return args.start_height + (deep - args.start_height) \
                    * (t - args.settle_s) / t_leg
            if t < t_top:                                    # now continuous UP
                return deep + (args.start_height - deep) \
                    * (t - t_bot) / t_leg
            return args.start_height

        label = (f"chain {args.start_height:.2f}->{deep:.2f}"
                 f"->{args.start_height:.2f}")
        res, tr = run((0.0, 0.0, 0.0), hfn, total, label, args.start_height,
                      capture=True, ascent_aware=True)
        res["stand_height"] = args.start_height
        res["z_min"] = (float(np.min(tr["z"])) if tr["z"].size
                        else float("nan"))
        res["recovered"] = ((not res["fall"])
                            and abs(res["z_settle"] - args.start_height)
                            <= args.hold_tol)
        res["held"] = res["recovered"]
        results.append(res)
        traces["chained"] = tr
        print(row(res))
        warn_asym(res, "the final standing hold")
        ok &= res["held"]

        # per-rung crossings, both legs
        # the reference just passes by
        # z is the lagging body,
        # NOT a settled height
        print("\nchained per-rung crossings (momentary, NOT settled; z is the "
              "lagged body, not the reference):")
        print(f"{'rung':>6} {'z_down':>7} {'t_down':>7} {'z_up':>7} {'t_up':>7}")
        ht = np.asarray(tr["h"])
        zt = np.asarray(tr["z"])
        tt = np.asarray(tr["t"])
        # one step of slack: the turning
        # point can land BETWEEN sim steps
        # equality would miss it
        tol_h = max(1e-9, slew * sim_dt)
        for rung in ladder:
            dn = np.flatnonzero((tt >= args.settle_s) & (ht <= rung + tol_h))
            up = np.flatnonzero((tt >= t_bot) & (ht >= rung - tol_h))
            i_dn = int(dn[0]) if dn.size else -1
            i_up = int(up[0]) if up.size else -1
            zd = zt[i_dn] if i_dn >= 0 else float("nan")
            td = tt[i_dn] if i_dn >= 0 else float("nan")
            zu = zt[i_up] if i_up >= 0 else float("nan")
            tu = tt[i_up] if i_up >= 0 else float("nan")
            print(f"{rung:6.3f} {zd:7.3f} {td:7.2f} {zu:7.3f} {tu:7.2f}")
    else:
        h_end = min(ladder)
        total = args.settle_s + args.ramp_s + max(args.measure_s, 2.0)

        def hfn(t):
            if t < args.settle_s:
                return args.start_height
            u = min(1.0, (t - args.settle_s) / args.ramp_s)
            return args.start_height + (h_end - args.start_height) * u
        res, tr = run((0.0, 0.0, 0.0), hfn, total,
                      f"ramp {args.start_height:.2f}->{h_end:.2f}", h_end,
                      capture=True)
        results.append(res)
        traces["ramped"] = tr
        print(row(res))
        warn_asym(res, "the deepest hold")
        ok &= res["held"]

        # snapshots per rung
        print("\nramp snapshots (momentary at each rung crossing, not settled):")
        print(f"{'cmd_h':>6} {'z':>6} {'kneL':>7} {'kneR':>7} {'dKnee':>6}")
        denom = args.start_height - h_end
        for rung in ladder:
            frac = 0.0 if denom <= 1e-9 else (args.start_height - rung) / denom
            i = min(int((args.settle_s + frac * args.ramp_s) / sim_dt), len(tr["z"]) - 1)
            k1, k2 = tr["knee_L"][i], tr["knee_R"][i]
            print(f"{rung:6.3f} {tr['z'][i]:6.3f} {k1:7.3f} {k2:7.3f} {abs(k1 - k2):6.3f}")

    # --- artifacts we leave behind ---
    # save one trajectory:
    # ramped/chained, or for recover
    # the DEEPEST rung — it tells most
    primary = traces.get("ramped") or traces.get("chained")
    if primary is None and primary_label is not None:
        primary = traces.get(primary_label)
    if primary is None:
        primary = next(iter(traces.values()), None)
    has_qpos = primary is not None and primary["qpos"] is not None
    if has_qpos:
        npz = os.path.join(args.out, f"trajectory_{args.mode}.npz")
        np.savez_compressed(npz, qpos=primary["qpos"], t=primary["t"],
                            height_cmd=primary["h"])
        which = (f", deepest rung {primary_label}" if args.mode == "recover"
                 and primary_label else "")
        print(f"trajectory: {npz} ({len(primary['qpos'])} steps, qpos 36-D"
              f"{which})")
    frames = [f for tr in traces.values() for f in (tr.get("frames") or [])]
    if frames:
        png_dir = os.path.join(args.out, "png")
        os.makedirs(png_dir, exist_ok=True)
        try:
            import imageio.v2 as imageio
            for i, img in enumerate(frames):
                imageio.imwrite(os.path.join(png_dir, f"frame_{i:04d}.png"), img)
            print(f"png: {png_dir} ({len(frames)} frames)")
            mp4 = os.path.join(args.out, f"height_sweep_{args.mode}.mp4")
            try:
                import imageio_ffmpeg  # noqa: F401  (ffmpeg backend present)
                with imageio.get_writer(mp4, fps=args.fps) as w:
                    for img in frames:
                        w.append_data(img)
                print(f"video: {mp4} ({len(frames)} frames @ {args.fps} fps)")
            except Exception as exc:  # noqa: BLE001
                print(f"video: mp4 skipped ({exc}) — PNG sequence in {png_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"video: PNG write failed ({exc})")
    # --- verdict: say EXACTLY what got judged, so it's falsifiable ---
    # one segment per judged rung
    # stepped/inband: every rung
    # ramped only the endpoint
    # momentary crossings
    # the old text claimed "every rung
    # held" and oversold
    # a one-height test
    judged = [float(r["h_cmd"]) for r in results]
    if args.mode == "ramped":
        crossed = [h for h in ladder if all(abs(h - j) > 1e-6 for j in judged)]
        scope = (f"ramped transient: JUDGED {len(judged)} of {len(ladder)} "
                 f"ladder rungs (settled endpoints only: "
                 f"{', '.join(f'{h:.3f}' for h in judged)}); "
                 f"crossed-but-NOT-judged: "
                 + (", ".join(f"{h:.3f}" for h in crossed) if crossed else "none"))
    elif args.mode == "recover":
        rec = [r for r in results if r["recovered"]]
        rec_h = ", ".join(f"{r['h_cmd']:.3f}" for r in rec) if rec else "none"
        scope = (f"recover: JUDGED {len(judged)} of {len(ladder)} ladder rungs "
                 f"(per rung: slew down to the rung, settle, slew back to "
                 f"standing {args.start_height:.3f} m at the writer slew "
                 f"{slew:.2f} m/s, settle); RECOVERED {len(rec)}/{len(judged)}: "
                 f"{rec_h}")
    elif args.mode == "chained":
        crossed = [h for h in ladder if all(abs(h - j) > 1e-6 for j in judged)]
        scope = (f"chained continuous down-then-up: JUDGED {len(judged)} of "
                 f"{len(ladder)} ladder rungs (settled endpoint only: the final "
                 f"standing return, target {args.start_height:.3f} m); "
                 f"crossed-but-NOT-judged (momentary, both legs): "
                 + (", ".join(f"{h:.3f}" for h in crossed) if crossed else "none"))
    else:
        scope = (f"{args.mode}: JUDGED {len(judged)} of {len(ladder)} ladder "
                 f"rungs (all settled): "
                 + ", ".join(f"{h:.3f}" for h in judged))
    metrics = {
        "mode": args.mode, "ladder": ladder, "start_height": args.start_height,
        "gates": {"fall_drop": args.fall_drop, "max_tilt": args.max_tilt,
                  "min_abs_z": args.min_abs_z, "hold_tol": args.hold_tol},
        # ascent modes gate on the
        # running min of height_cmd
        # descent gate untouched (docstring)
        # recorded so you can audit it
        "ascent_aware_gate": args.mode in ("recover", "chained"),
        "rungs": results, "render_ok": render_ok,
        "judged_rungs": judged, "judged_count": len(judged),
        "ladder_count": len(ladder), "claim_scope": scope,
        "writer_band": {"lo": band_lo, "hi": band_hi, "slew_rate": slew},
    }
    if args.mode == "recover":
        metrics["recovered_count"] = sum(1 for r in results if r["recovered"])
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # per-mode sidecar:
    # metrics.json is one shared name
    # every mode overwrites, so artifacts
    # would look mixed
    with open(os.path.join(args.out, f"metrics_{args.mode}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if renderer is not None:
        try:
            renderer.close()
        except Exception:  # noqa: BLE001
            pass
    print()
    print("claim scope — " + scope)
    if ok:
        if args.mode == "recover":
            print(f"mj_height_sweep GREEN — every one of the {len(judged)} "
                  f"JUDGED rung(s) was reached AND recovered to standing "
                  f"{args.start_height:.3f} m ({args.mode} mode; descent gate "
                  f"unchanged, ascent gate running-min; NOT weakened)")
        elif args.mode == "chained":
            print(f"mj_height_sweep GREEN — the continuous down-then-up run "
                  f"returned to standing {args.start_height:.3f} m within "
                  f"{args.hold_tol:.2f} m ({args.mode} mode; descent gate "
                  f"unchanged, ascent gate running-min; NOT weakened)")
        else:
            print(f"mj_height_sweep GREEN — every one of the {len(judged)} "
                  f"JUDGED rung(s) held at its commanded height "
                  f"({args.mode} mode; gate NOT weakened)")
    else:
        bad = [f"h={r['h_cmd']:.3f}" for r in results if not r["held"]]
        verb = ("not recovered to standing" if args.mode == "recover"
                else "not held")
        print(f"mj_height_sweep FAILED — judged rungs {verb}:",
              ", ".join(bad))
        for r in results:
            if not r["held"]:
                if r["fall"]:
                    why = r["fall_reason"]
                elif args.mode == "recover":
                    why = (f"z_rung={r.get('z_rung_settle', float('nan')):.3f} "
                           f"(reached={r.get('rung_reached')}), "
                           f"z_final={r['z_settle']:.3f} "
                           f"(recovered={r.get('recovered')})")
                else:
                    why = (f"|z_settle-cmd|={abs(r['z_settle'] - r['h_cmd']):.3f}"
                           f" > hold_tol={args.hold_tol:.2f}")
                print(f"  h={r['h_cmd']:.3f}: z_settle={r['z_settle']:.3f} "
                      f"({why})")
    arts = ("metrics.json" + (", trajectory_*.npz" if has_qpos else "")
            + (", png/, height_sweep_*.mp4" if frames else ""))
    print(f"artifacts: {args.out} ({arts})")
    if not render_ok and want_render:
        print("render later (needs working offscreen GL): "
              "MUJOCO_GL=egl python3 tools/mj_height_sweep.py --mode ramped "
              f"--out {args.out}   |   or replay the .npz qpos with mujoco.Renderer")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
