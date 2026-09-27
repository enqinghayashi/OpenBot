#!/usr/bin/env python3
"""main_merged_writer.py — C3 merged whole-body writer for Stuart (G1 EDU).

ONE process owns rt/lowcmd for ALL 35 slots, from boot to shutdown. No phase
split: legs+waist (0-14) from the GR00T Balance/Walk ONNX policy at 50 Hz,
arms (15-28) from arm_guard pose logic at 250 Hz, gate joint 29 + trailing
30-34 held. Walk while holding the carry-tuck. No rclpy, no listener-mode
subscribers, no sport RPC. See docs/MERGED_WRITER.md.

Bring-up: flock -> MotionSwitcher CheckMode/ReleaseMode until name empty ->
first lowstate -> capture q35 -> base frame (mode 0x01, mode_pr=0,
mode_machine echoed) -> operator GATE prompt -> 120 ms coupling ramp -> loop.

Shutdown (SIGINT/SIGTERM/quit/watchdog): rc zero -> policy 1 s -> arm release
ramp (skipped with a warning if payload flag set) -> legs hold 1 s ->
ZeroCmd -> SHUTDOWN COMPLETE -> handoff ladder printed.

v2b socket (Stage 4): thread-per-connection UNIX bridge at cfg sock_path,
one reply line per request (DONE/ERROR/Q:/STATE:). New verbs: arm <14f>
[duration_s], freeze, q, armstate, hold_rc on|off. DONE means the slew was
STARTED (the client polls armstate for completion — the guarded insert
monitors q mid-slew, so a blocking DONE would serialize the monitor out).
rc_hold clamps the RC sticks (pick interlock) exactly like _shutting_down.

Runtime height lane (2026-09-10; REPO-SIDE ONLY, NOT deployed): the `height
<m> [duration_s]` verb rate-limits the obs height reference (obs_builder f[3])
so the policy tracks a squat; height 0.74 restores standing.  It is NOT
orthogonal to cmd the way this docstring used to claim.  The pick engine sets
rc_hold during arm motion, which forces rc_live = False in the precedence
block, which gates the height suspend off — so the live RC stick is
UNREACHABLE for the whole arm motion and CANNOT abort a squat while the engine
holds the body.  A stick push during such a squat is a PAUSE, not a cancel:
the writer stashes the target and the ORIGINAL deadline and the resume restores
BOTH.  It carries a monotonic deadline watchdog (recovery = standing, not zero)
and shutdown ascends the legs before the limp frame.  See the CORRECTED block
in _tick, marker "CORRECTED 2026-09-27 (D2)".
"""

import argparse
import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------- SONIC lane
# gain and joint tables come from sonic_core.
# no onnxruntime is used; an import failure forces the lane off.
_SONIC_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sonic")
if os.path.isdir(_SONIC_DIR):
    sys.path.insert(0, _SONIC_DIR)
try:
    import sonic_core as SC
except Exception as _e:            # noqa: BLE001 — degrade, never raise
    SC = None
    print(f"[sonic] sonic_core import FAILED ({_e!r}) — SONIC lane FORCED OFF",
          flush=True)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from arm_guard import ArmGuard
from obs_builder import ObsBuilder
from policy_runner import PolicyRunner
from rc_target import RCTarget
from state_reader import StateReader

PosStopF = 2146000000.0    # pos-stop sentinel
VelStopF = 16000.0         # vel-stop sentinel
MODE_ENABLE = 1            # enable mode

# stance: ankle roll per hip roll, holding the sole tilt flat.
# FK: -0.907/-0.889 crouch, -0.831/-0.829 stand, -0.799/-0.799 squat.
# A 6 deg splay holds ~0.5 deg sole error. Mid-range default.
STANCE_ANKLE_RATIO = -0.85

# ---------------------------------------------------------------- step lane
# every step constant is derived from the URDF forward kinematics,
# not from body measurements.
STEP_SEP_LEVER_M = 0.4955        # hip->ankle, 8.65 mm/deg
                                 # W = 0.4955*(sin d1 - sin d7)
STEP_COM_CM_PER_DEG = 0.6856     # same-sign, cm/deg
STEP_KNEE_LEVER_M = 0.109        # N = |tau_knee|/0.109
STEP_LIFT_DZ_CM_PER_DEG = 0.2333  # K -> foot rise, cm/deg
STEP_LIFT_HIP_PITCH = -0.472     # d_pol[0]/d_pol[6] per K
STEP_LIFT_ANKLE_PITCH = -0.528   # d_pol[4]/d_pol[10] per K
STEP_ANKLE_STOP_DEG = 15.0       # 15 deg stop, URDF
STEP_ANKLE_BASE_DEG = 1.0        # +1.0, both legs
STEP_TWO_STEP_MAX_CM = 27.4      # ceiling 27.4 cm

# honesty suffix: the writer has no body-height sensor, so every [step]
# body-state line says the value is writer-reported, not sensor-measured.
# Appended in exactly one place.
STEP_HONESTY_SUFFIX = ("writer-reported, NOT sensor-measured — "
                       "no body-height sensor exists")

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "config", "merge_config.yaml")


def kb_safe_sleep(sec):
    """Sleep that survives SIGINT. Inside shutdown a second ^C must never
    abort the crash-safe tail (the final ZeroCmd frame + lockfile removal +
    sport handoff must always complete). With the Python SIGINT handler
    installed a terminal INT normally just runs the handler and resumes the
    sleep, but a KeyboardInterrupt escaping from any path must not kill the
    tail either."""
    try:
        time.sleep(sec)
    except KeyboardInterrupt:
        print("[shutdown] KeyboardInterrupt swallowed — shutdown continues "
              "(ZeroCmd frame + lockfile removal + handoff complete)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="C3 merged whole-body writer")
    p.add_argument("--stage", type=int, default=1, choices=[0, 1, 2, 3, 4],
                   help="0=hold-only ownership probe (no policy); >=1 policy active")
    p.add_argument("--onnx-dir", default=os.environ.get("MERGE_ONNX_DIR"),
                   help="override ONNX dir (default: <pkg>/policy)")
    p.add_argument("--no-rc", action="store_true", help="disable rc (cmd stays zero)")
    p.add_argument("--interface", default=None, help="DDS interface (default: eth0)")
    p.add_argument("--config", default=CFG_PATH, help="config yaml path")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


class MergedWriter:
    # ------------------------------------------------------------ SONIC lane
    # lane defaults; keep in sync with _init_sonic_state. An object built
    # with object.__new__ then reads the lane as off instead of raising
    # AttributeError on the 250 Hz path.
    sonic_lane_enabled = False
    sonic_mode = 'off'
    sonic_armed = False
    sonic_b = 0.0
    sonic_target = None
    sonic_last_rx = 0.0
    sonic_rx = 0
    sonic_rej = 0
    sonic_seq = -1
    sonic_warm = 0
    sonic_last_good_t = 0.0
    sonic_faulted = False
    sonic_fault_cause = "none"
    sonic_prev_blend = 0.0
    sonic_has_frame = False
    sonic_log_t = 0.0
    sonic_overrun_head = 0
    sonic_overrun_base = 0
    sonic_kp_boot = None
    sonic_kd_boot = None
    # gate probe: ''=off
    sonic_gate_probe_dir = ""
    sonic_gate_probe_max = 512
    sonic_gate_probe_fh = None
    sonic_gate_probe_path = None
    sonic_gate_probe_n = 0
    sonic_gate_dev = None
    # STEP defaults; keep in sync with _init_step_state so an
    # uninitialised object cannot crash a tick.
    step_shift_deg_max = 0.0
    step_cm_max = 0.0
    step_slew_deg_s = 8.0
    step_lift_slew_deg_s = 6.0
    step_clearance_cm = 3.0
    step_knee_max_deg = 20.0
    step_ankle_margin_deg = 2.0
    step_min_shift_deg = 0.0
    step_unload_tau_x = 2.0
    step_hold_s = 2.0
    step_tilt_deg = 8.0
    step_shift_deg = 0.0
    step_shift_target = 0.0
    step_c1 = 0.0
    step_c7 = 0.0
    step_c1_t = 0.0
    step_c7_t = 0.0
    step_splay_l = 0.0
    step_splay_r = 0.0
    step_splay_l_t = 0.0
    step_splay_r_t = 0.0
    step_k = 0.0
    step_k_t = 0.0
    step_phase = "IDLE"
    step_leg = ""
    step_cm_req = 0.0
    step_cm_ach = 0.0
    step_w_m = 0.0
    step_tau0 = None
    step_dtau = None
    step_hold_t = None
    step_go_pending = False
    step_abort_reason = ""
    step_tilt = None
    step_log_t = 0.0
    _step_last_key = None
    _stance_base_policy = None
    _stance_base_obs = None

    def __init__(self, args):
        self.args = args
        with open(args.config) as f:
            self.cfg = yaml.safe_load(f)

        if args.onnx_dir:
            self.onnx_dir = args.onnx_dir
        else:
            self.onnx_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                self.cfg["onnx_dir"])
        self.onnx_dir = os.path.abspath(self.onnx_dir)

        self.hz = int(self.cfg["control_hz"])
        self.dt = 1.0 / self.hz
        self.decimation = int(self.cfg["policy_decimation"])
        self.ramp_s = float(self.cfg["coupling_ramp_s"])
        self.mode_machine_cfg = int(self.cfg["mode_machine"])

        self.policy_enabled = args.stage >= 1
        self.rc_enabled = not args.no_rc
        self.verbose = args.verbose
        self.payload = False          # console: payload on/off
        self.rc_hold = False          # pick interlock: sticks clamped
        self.cmd_auto = np.zeros(3, dtype=np.float32)   # latched cmd
        self.cmd_auto_active = False  # rc ignored
        self.cmd_auto_deadline = 0.0  # expiry watchdog
        self._dispatch_lock = threading.Lock()  # guards arm writes
        self._init_height_state()     # height (inert by default)
        self._init_sonic_state()      # SONIC (opt-in)
        self._init_stance_state()     # stance (shut)
        self._init_step_state()       # step (shut)

        self.state = "INIT"
        self._shutdown = False
        self._shutting_down = False
        self._shutdown_cause = "none"
        self.t0 = None                # loop start
        self.loop_count = 0
        self.policy_ticks = 0
        self.prev_tick = None         # for drop count
        self._overruns = 0
        self._dropped = 0
        self.rc_cmd_zero = np.zeros(3, dtype=np.float32)

        self.lowcmd = None
        self.lowcmd_publisher = None
        self.lowstate = None          # last lowstate
        self.captured_q = None        # 35-D boot pose

        self.policy = None
        self.rc = None
        self.arm = None
        self.obs = None
        self.leg_target = None        # policy target
        self.policy_target = None     # SONIC blend input

        # arm shutdown slew: 0.8 rad/s, faster than the 0.3 rad/s pose
        # slew, with a 0.5 s minimum.
        self.shutdown_slew_rate = float(self.cfg.get("arm_shutdown_slew_rate", 0.8))
        self.shutdown_slew_min = float(self.cfg.get("arm_shutdown_slew_min_s", 0.5))

        self._simulate_fatal = False   # test hook

    # ------------------------------------------------------------ bring-up

    def acquire_lock(self):
        self.lock_fd = open(self.cfg["lockfile"], "w")
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"lock held: {self.cfg['lockfile']} — "
                               "another writer is running")
        self.lock_fd.write(str(os.getpid()))
        self.lock_fd.flush()

    def release_mode(self):
        """MotionSwitcher RELEASED (debug mode, name empty)."""
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        while result["name"]:
            print(f"[bring-up] mode '{result['name']}' active — ReleaseMode")
            msc.ReleaseMode()
            time.sleep(1.0)
            status, result = msc.CheckMode()
        print(f"[bring-up] MotionSwitcher released (name empty, status {status})")

    def init_lowcmd(self):
        self.lowcmd = unitree_hg_msg_dds__LowCmd_()
        self.lowcmd.mode_machine = self.mode_machine_cfg
        self.lowcmd.mode_pr = 0  # series control
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        for i in range(35):
            self.lowcmd.motor_cmd[i].mode = MODE_ENABLE
            self.lowcmd.motor_cmd[i].q = self.captured_q[i]
            self.lowcmd.motor_cmd[i].dq = 0.0
            self.lowcmd.motor_cmd[i].tau = 0.0
        self.crc = CRC()

    def fill_gains(self):
        """Write the verified gain tables into motor_cmd 0..34 (once at boot)."""
        leg_kp = np.array(self.cfg["leg_kp"], dtype=np.float32)
        leg_kd = np.array(self.cfg["leg_kd"], dtype=np.float32)
        if float(self.cfg.get("waist_pitch_kd_bias", 0.0)):
            leg_kd = leg_kd.copy()
            leg_kd[14] += float(self.cfg["waist_pitch_kd_bias"])  # waist_pitch
        for i in range(15):
            self.lowcmd.motor_cmd[i].kp = leg_kp[i]
            self.lowcmd.motor_cmd[i].kd = leg_kd[i]
        arm_kp = np.array(self.cfg["arm_kp"], dtype=np.float32)
        arm_kd = np.array(self.cfg["arm_kd"], dtype=np.float32)
        for j, i in enumerate(range(15, 29)):
            self.lowcmd.motor_cmd[i].kp = arm_kp[j]
            self.lowcmd.motor_cmd[i].kd = arm_kd[j]
        self.lowcmd.motor_cmd[29].kp = float(self.cfg["gate_kp"])
        self.lowcmd.motor_cmd[29].kd = float(self.cfg["gate_kd"])
        for i in range(30, 35):
            self.lowcmd.motor_cmd[i].kp = float(self.cfg["trailing_kp"])
            self.lowcmd.motor_cmd[i].kd = float(self.cfg["trailing_kd"])
        # SONIC boot gains, snapshotted after the waist bias override:
        # blending b=0 restores them bit-exactly. Filled here because
        # the motor commands already exist.
        self.sonic_kp_boot = np.array(
            [self.lowcmd.motor_cmd[i].kp for i in range(15)], dtype=np.float64)
        self.sonic_kd_boot = np.array(
            [self.lowcmd.motor_cmd[i].kd for i in range(15)], dtype=np.float64)

    def gate_prompt(self):
        print("\n=== GATE: gantry on? spotter ready? physical E-stop armed? ===")
        if self.policy_enabled:
            print("=== POLICY IS ACTIVE in this run (stage >= 1) ===")
        if sys.stdin.isatty():
            input("Press Enter to arm the writer and start the coupling ramp... ")
        else:
            # no tty: the launcher already gated, so arm immediately.
            print("[gate] stdin not a tty (launcher-gated) — arming immediately")

    # ------------------------------------------------------------ console

    def start_console(self):
        def loop():
            while not self._shutdown:
                try:
                    line = input("merge> ")
                except (EOFError, OSError):
                    break
                try:
                    self.console_command(line.strip())
                except Exception as e:
                    print(f"[console] {e}")
        threading.Thread(target=loop, daemon=True).start()

    # --------------------------------------------------- socket server

    def start_socket_server(self):
        """UNIX-socket request bridge with one-line replies (v2b).

        Thread-per-connection (grip_server pattern, already validated on
        Stuart): the pick engine polls `q` at 50 Hz while an `arm` slew is
        still in flight on another connection. One line per request, one
        reply line per request (DONE = slew STARTED; completion is observed
        via `armstate` — the guarded insert monitors q mid-slew, so a
        blocking DONE would serialize the monitor out):
          arm <14 floats> [duration_s]   DONE / ERROR
          freeze | q | armstate | hold_rc on|off
          cmd_set off|vx vy omega [dur]  height <m> [dur] | height off
          stance <deg> | stance off      DONE (symmetric hip-roll splay)
          step shift <deg> | step go l|r <cm> | step off | step status
          payload on|off rc0 status quit
          sonic <seq> <age_ms> <q0..q14> SONIC:f <seq> / ERROR (18-token 50 Hz frame)
          sonic on|off|status            SONIC:on <b> / SONIC:off / SONIC:status ...
          sonic arm | sonic reset         ERROR:<verb> is console-only
          estop                          DONE (SONIC lane force-released)
        Dispatched through the same actions as console_command, so socket
        and console stay in lockstep. A thread wakes only on a request —
        not a DDS callback, so the GIL rule is untouched.

        Log suppression (spec §2.3): the `[sock] <line>` echo is dropped ONLY
        when the reply starts with `SONIC:f ` (the 50 Hz steady-state ack) or
        is byte-identical to the previous reply on THIS connection (idempotent
        re-ack).  Every ERROR: and every state-change reply still prints.
        """
        path = self.cfg.get("sock_path")
        if not path:
            print("[sock] no sock_path in config — bridge disabled")
            return
        try:
            os.unlink(path)
        except OSError:
            pass

        def handle(conn):
            conn_file = conn.makefile("r", encoding="utf-8")
            prev_reply = None       # per-conn reply
            try:
                for raw in conn_file:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        reply = self.handle_socket_line(line, conn)
                    except Exception as e:
                        reply = f"ERROR:{e}"
                    # suppress the SONIC:f ack and a byte-identical repeat;
                    # an ERROR always prints.
                    if not (reply.startswith("SONIC:f ") or reply == prev_reply):
                        print(f"[sock] {line}")
                    prev_reply = reply
                    try:
                        conn.sendall((reply + "\n").encode())
                    except OSError:
                        break
            except OSError:
                pass
            finally:
                conn_file.close()
                conn.close()

        def loop():
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(path)
            srv.listen(4)
            print(f"[sock] listening on {path}")
            while not self._shutdown:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    break
                threading.Thread(target=handle, args=(conn,), daemon=True).start()
            srv.close()
        threading.Thread(target=loop, daemon=True).start()

    def handle_socket_line(self, line, conn=None):
        """One socket request -> one reply line (DONE / ERROR / Q: / STATE: /
        SONIC:).

        `conn` is the connection the request arrived on.  The spec §2.3
        log-suppression rule keys on the PREVIOUS REPLY ON THAT CONNECTION, so
        the connection identity is threaded through here; the `[sock] <line>`
        echo itself is emitted by the connection loop in
        `start_socket_server`, which owns that per-connection previous reply.
        """
        parts = line.split()
        verb = parts[0] if parts else ""
        if not verb:
            return "ERROR:empty request"
        now = time.perf_counter()
        if verb in ("tuck", "carry", "ready", "lift", "neutral", "home"):
            pose = "CARRY_TUCK" if verb in ("tuck", "carry") else verb.upper()
            with self._dispatch_lock:
                self.arm_tau[:] = 0.0   # no FF on slews
                self.arm.slew_to(pose, now)
            return "DONE"          # STARTED; poll armstate
        if verb == "arm":
            if len(parts) < 15:
                return "ERROR:arm needs 14 floats"
            try:
                target = np.array([float(x) for x in parts[1:15]], dtype=np.float32)
            except ValueError:
                return "ERROR:arm needs 14 floats"
            duration = None
            tau = None
            if len(parts) >= 16:
                try:
                    duration = float(parts[15])
                except ValueError:
                    return "ERROR:arm duration must be a float"
            if len(parts) >= 30:
                # gravity feed-forward for the 14 arm joints (15-28),
                # applied at 250 Hz.
                try:
                    tau = np.array([float(x) for x in parts[16:30]],
                                   dtype=np.float64)
                except ValueError:
                    return "ERROR:arm tau must be 14 floats"
            with self._dispatch_lock:
                self.arm_tau[:] = tau if tau is not None else 0.0
                self.arm.slew_to_vector(target, now, duration)
            return "DONE"          # STARTED; poll
        if verb == "freeze":
            with self._dispatch_lock:
                self.arm.freeze(time.perf_counter())
            return "DONE"
        if verb == "q":
            st = self.lowstate
            if st is None:
                return "ERROR:no lowstate"
            return "Q:" + " ".join(f"{v:.6f}" for v in st["q"][15:29])
        if verb == "armstate":
            a = self.arm
            return f"STATE: {a.state} {1 if a._T > 0.0 else 0} {a._T:.3f}"
        if verb == "hold_rc":
            if len(parts) != 2 or parts[1] not in ("on", "off"):
                return "ERROR:hold_rc on|off"
            self.rc_hold = (parts[1] == "on")
            print(f"[rc] hold_rc={self.rc_hold}")
            return "DONE"
        if verb == "payload":
            if len(parts) != 2 or parts[1] not in ("on", "off"):
                return "ERROR:payload on|off"
            self.payload = (parts[1] == "on")
            print(f"[arm] payload={self.payload}")
            return "DONE"
        if verb == "rc0":
            if self.rc:
                self.rc.zero()
            return "DONE"
        if verb == "cmd_set":
            err = self._apply_cmd_set(parts)
            if err:
                return f"ERROR:{err}"
            print(f"[cmd] {self._cmd_auto_status()}")
            return "DONE"
        if verb == "height":
            err = self._apply_height(parts)
            if err:
                return f"ERROR:{err}"
            print(f"[height] {self._height_status()}")
            return "DONE"
        if verb == "stance":
            err = self._apply_stance(parts)
            if err:
                return f"ERROR:{err}"
            print(f"[stance] {self._stance_status()}")
            return "DONE"
        if verb == "step":
            # STEP:<frag> reply
            # else DONE/ERROR
            if len(parts) == 2 and parts[1] == "status":
                return f"STEP:{self._step_status()}"
            err = self._apply_step(parts)
            if err:
                return f"ERROR:{err}"
            print(f"[step] {self._step_status()}")
            return "DONE"
        if verb == "sonic":
            # console-only verb forms are handled in _apply_sonic;
            # anything else is an 18-token frame.
            if len(parts) == 1:
                return f"SONIC:status {self._sonic_status()}"
            if parts[1] in ("arm", "on", "off", "reset", "status"):
                return self._apply_sonic(parts, from_console=False)
            # 18-token frame. Broad except: a ValueError here would escape
            # run()'s RuntimeError-only handler and kill the writer.
            try:
                if len(parts) != 18:
                    self.sonic_rej += 1
                    return "ERROR:sonic frame needs 18 tokens"
                vals = [float(x) for x in parts[1:18]]      # 17 numeric fields
                seq = int(vals[0])
                age_ms = vals[1]
                q = np.array(vals[2:17], dtype=np.float64)
                if not np.isfinite(q).all():
                    self.sonic_rej += 1
                    return "ERROR:non-finite q"
                if np.max(np.abs(q)) > self.sonic_target_abs_max:
                    self.sonic_rej += 1
                    return "ERROR:q out of range"
                prev = self.sonic_target
                if prev is not None and \
                        np.max(np.abs(q - prev)) > self.sonic_frame_step_max:
                    self.sonic_rej += 1
                    return "ERROR:frame step too large"
                if seq <= self.sonic_seq:
                    self.sonic_rej += 1
                    return "ERROR:stale seq"
                # negated test: NaN fails both <= and >, so a plain `>` would
                # read NaN as fresh forever and the stale revoke would die.
                if not (age_ms <= self.sonic_stale_ms):
                    self.sonic_rej += 1
                    return "ERROR:state age too old"
                with self._dispatch_lock:
                    # warmup counts a consecutive run of accepted frames;
                    # a gap over stale_ms restarts it, as in the live lane.
                    now = time.monotonic()
                    self._sonic_warm_tick(now)
                    # rebind, not [:], so the 250 Hz tick only sees one array.
                    self.sonic_target = q
                    self.sonic_last_rx = now
                    self.sonic_seq = seq
                    self.sonic_rx += 1
                    self.sonic_has_frame = True
                return f"SONIC:f {seq}"
            except Exception:
                self.sonic_rej += 1
                return "ERROR:sonic frame rejected"
        if verb == "estop":
            # force-release only: no shutdown call, so the motors never
            # go limp mid-walk.
            self._sonic_revoke("estop", force=True)
            return "DONE"
        if verb == "simulate_fatal":
            self._simulate_fatal = True
            print("[fatal] SIMULATED FATAL armed")
            return "DONE"
        if verb == "status":
            key = self.policy.last_key if self.policy else "no-policy"
            a = self.arm
            return (f"STATE: {a.state} {1 if a._T > 0.0 else 0} {a._T:.3f} "
                    f"policy={key} rc_hold={int(self.rc_hold)} "
                    f"payload={int(self.payload)} "
                    f"{self._cmd_auto_status()} {self._height_status()} "
                    f"{self._stance_status()} {self._sonic_status()} {self._step_status()}")
        if verb == "quit":
            self.request_shutdown("socket quit")
            return "DONE"
        return "ERROR:unknown command"

