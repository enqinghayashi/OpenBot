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

    def _apply_cmd_set(self, parts):
        """cmd_set parse+apply, shared by socket and console (same dispatch
        pattern as arm/hold_rc in both handlers). Returns an error message or
        None on success; mutates the cmd_auto state on success:
          cmd_set off                    deactivate + zero
          cmd_set vx vy omega            activate, 1.0 s
          cmd_set vx vy omega duration   activate, duration > 0
        """
        if len(parts) == 2 and parts[1] == "off":
            self.cmd_auto_active = False
            self.cmd_auto[:] = 0.0
            return None
        if len(parts) not in (4, 5):
            return "cmd_set off | cmd_set vx vy omega [duration_s]"
        try:
            vals = np.array([float(x) for x in parts[1:4]], dtype=np.float32)
        except ValueError:
            return "cmd_set needs 3 floats"
        duration = 1.0
        if len(parts) == 5:
            try:
                duration = float(parts[4])
            except ValueError:
                return "cmd_set duration must be a float"
        if not np.isfinite(vals).all() or not np.isfinite(duration):
            return "cmd_set values must be finite"
        if duration <= 0.0:
            return "cmd_set duration must be > 0"
        self.cmd_auto[:] = vals
        self.cmd_auto_active = True
        self.cmd_auto_deadline = time.monotonic() + duration
        return None

    def _cmd_auto_status(self):
        """'auto=0' or 'auto=1 cmd=(vx,vy,om) rem=..s' — inline-field fragment
        for the three status surfaces (console, _status_line, STATE:)."""
        if not self.cmd_auto_active:
            return "auto=0"
        rem = max(self.cmd_auto_deadline - time.monotonic(), 0.0)
        c = self.cmd_auto
        return (f"auto=1 ({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) "
                f"rem={rem:.1f}s")

    # ------------------------------------------------- runtime height lane
    # drives the obs height reference (f[3]), not a velocity.
    # not orthogonal to cmd: rc_hold forces rc_live False, so the stick
    # can only suspend a squat, never abort it.
    # recovery returns to height_stand, not to zero.

    def _init_height_state(self):
        """Height-lane state + the POLICY-MEASURED standing-pose reference.

        Every key is read with `.get(..., <safe default>)`, matching
        obs_builder.py:38-39 — an older config that predates the lane must not
        raise KeyError at construction (FINDING 6).  The band defaults to the
        nominal height, i.e. a no-op band, so the lane can only ever hold
        standing.
        """
        self.height_stand = float(self.cfg.get("height_cmd", 0.74))   # nominal height
        self.height_min = float(self.cfg.get("height_cmd_min", self.height_stand))
        self.height_max = float(self.cfg.get("height_cmd_max", self.height_stand))
        if self.height_min > self.height_max:   # inverted ok
            self.height_min, self.height_max = self.height_max, self.height_min
        self.height_slew_rate = float(self.cfg.get("height_slew_rate", 0.15))
        self.height_hold_s = float(self.cfg.get("height_hold_s", 5.0))
        self.height_ascend_timeout = float(
            self.cfg.get("height_ascend_timeout_s", 3.0))
        self.height_ascend_tol = float(
            self.cfg.get("height_ascend_tol_rad", 0.25))
        self.height_rc_override = float(
            self.cfg.get("height_rc_override_thresh", 0.1))
        # hysteresis: suspend at the override threshold, resume at half.
        self.height_rc_resume = float(
            self.cfg.get("height_rc_resume_thresh",
                         0.5 * self.height_rc_override))
        self.height_stand_eps = 1e-3
        self.height_ref = self.height_stand      # obs f[3] ref
        self.height_target = self.height_stand   # clamped request
        self.height_active = False               # latch + deadline
        self.height_deadline = 0.0
        # pause, not cancel: the target and the original deadline
        # are stowed here.
        self.height_suspended = False
        self.height_suspend_target = self.height_stand
        self.height_suspend_deadline = 0.0
        self._height_log_t = 0.0
        # reference standing q (0-11) from the policy's settled pose. The
        # YAML default_angles miss that knee angle by 0.44 rad, past tol
        # 0.25; policy_stand_q_legs overrides when present.
        d = np.array(self.cfg["default_angles"], dtype=np.float32)
        stand = d.copy()
        measured = self.cfg.get("policy_stand_q_legs")
        if measured is not None:
            arr = np.array(measured, dtype=np.float32)
            if arr.shape[0] >= 12:
                stand[0:12] = arr[0:12]
        self.height_stand_q = np.concatenate(
            [stand, np.zeros(35 - stand.shape[0], dtype=np.float32)])

    def _apply_height(self, parts):
        """height parse+apply, shared by socket and console (the same lockstep
        split that keeps cmd_set identical at both dispatch sites).  Returns an
        error message or None on success:
          height <m>                 clamp+set ref, hold height_hold_s
          height <m> <duration_s>    clamp+set ref, hold duration
          height off                 cancel now, return to standing
        `height 0.74` (nominal standing) also returns to standing — reversible
        at any time.  Values are clamped to [height_cmd_min, height_cmd_max] and
        _tick rate-limits the reference (height_slew_rate), so nothing steps.
        """
        if len(parts) == 2 and parts[1] == "off":
            self._return_to_standing("height off")
            return None
        if len(parts) not in (2, 3):
            return "height off | height <m> [duration_s]"
        try:
            val = float(parts[1])
        except ValueError:
            return "height needs a float"
        duration = self.height_hold_s
        if len(parts) == 3:
            try:
                duration = float(parts[2])
            except ValueError:
                return "height duration must be a float"
        if not np.isfinite(val) or not np.isfinite(duration):
            return "height values must be finite"
        if duration <= 0.0:
            return "height duration must be > 0"
        val = min(max(val, self.height_min), self.height_max)
        if abs(val - self.height_stand) <= self.height_stand_eps:
            self._return_to_standing("height standing")
            return None
        # new squat wins
        self.height_suspended = False
        self.height_target = val
        self.height_active = True
        self.height_deadline = time.monotonic() + duration
        return None

    def _return_to_standing(self, cause):
        """Clear the height latch; _tick slews height_ref back up to
        height_stand.  Used by `height off`, `height 0.74`, the watchdog, and
        every shutdown path.  Also drops any SUSPENDED squat (a suspend is only
        ever entered directly by the RC-override block in _tick, which stows its
        target before clearing the latch)."""
        if self.height_active:
            self._height_log(f"{cause} -> standing")
        self.height_active = False
        self.height_target = self.height_stand
        self.height_suspended = False

    def _height_log(self, msg, force=False):
        """Rate-limited (1 Hz) height-lane log — no assert, no exception.
        `force` bypasses the rate limit for the rare state transitions that
        must always be visible (RC suspend/resume, Finding 7)."""
        now = time.monotonic()
        if force or now - self._height_log_t >= 1.0:
            self._height_log_t = now
            print(f"[height] {msg}")

    def _height_status(self):
        """Inline status fragment, same shape as _cmd_auto_status."""
        if self.height_suspended:
            rem = max(self.height_suspend_deadline - time.monotonic(), 0.0)
            return (f"height={self.height_ref:.2f} "
                    f"SUSPENDED->{self.height_suspend_target:.2f} rem={rem:.1f}s")
        if not self.height_active:
            return f"height={self.height_ref:.2f}(stand {self.height_stand:.2f})"
        rem = max(self.height_deadline - time.monotonic(), 0.0)
        return (f"height={self.height_ref:.2f}->{self.height_target:.2f} "
                f"active rem={rem:.1f}s")

    # ---------------------------------------------------- additive stance lane
    # hip-roll splay: no deadline, no watchdog, no thread, no height
    # change. Sticky until `stance off` and never persisted; a stuck
    # splay leaves the robot wide, not tipped.

    def _init_stance_state(self):
        """Stance-lane state.  SAFE BY DEFAULT: the band ships SHUT, so a config
        that predates this lane cannot splay the legs and `stance <deg>` can
        only ever return to 0.  Arming is a deliberate config edit.

        Every key is read with `.get(..., <safe default>)` for the same reason
        _init_height_state does — an older config must not raise KeyError at
        construction."""
        self.stance_deg_max = float(self.cfg.get("stance_deg_max", 0.0))
        if not np.isfinite(self.stance_deg_max) or self.stance_deg_max < 0.0:
            self.stance_deg_max = 0.0     # NaN/neg = shut
        self.stance_ankle_ratio = float(
            self.cfg.get("stance_ankle_ratio", STANCE_ANKLE_RATIO))
        if not np.isfinite(self.stance_ankle_ratio):
            self.stance_ankle_ratio = STANCE_ANKLE_RATIO
        self.stance_deg = 0.0             # installed splay
        self._stance_base_policy = None   # boot copies, run()
        self._stance_base_obs = None
        self._stance_log_t = 0.0

    def _capture_stance_base(self):
        """Snapshot the boot `default_angles` encodings VERBATIM.  Every stance
        write is re-derived from these and never compounded, which is what makes
        the verb idempotent and makes `stance off` a BIT-EXACT restore."""
        if self.policy is None or self.obs is None:
            return
        self._stance_base_policy = self.policy.defaults.copy()
        self._stance_base_obs = self.obs.default_padded.copy()

    def _apply_stance(self, parts):
        """stance parse+apply, shared by socket and console (the same lockstep
        split that keeps height/cmd_set identical at both dispatch sites).
        Returns an error message or None on success:
          stance <deg>    symmetric outward hip-roll splay, clamped to band
          stance off      restore the boot stance (identical to `stance 0`)

        The band is [-stance_deg_max, +stance_deg_max] and ships SHUT at 0.0,
        so an unarmed writer refuses to splay.  Reversible at any time."""
        if len(parts) == 2 and parts[1] == "off":
            deg = 0.0
        elif len(parts) == 2:
            try:
                deg = float(parts[1])
            except ValueError:
                return "stance needs a float"
            if not np.isfinite(deg):
                return "stance value must be finite"
        else:
            return "stance off | stance <deg>"
        if self._stance_base_policy is None:
            return "stance unavailable: no policy (stage 0)"
        deg = min(max(deg, -self.stance_deg_max), self.stance_deg_max)
        self._write_stance(deg)
        return None

    def _write_stance(self, deg):
        """Install the stance offset into BOTH policy-side encodings.

        THE MECHANISM, as MEASURED.  policy_runner:50 is
        `target_q = action * action_scale + defaults`, and obs_builder:105 is
        `f[13:42] = q - default_padded`; both read the SAME config key, but
        through two INDEPENDENT np.array copies (policy_runner:19,
        obs_builder:43-46).  Shift BOTH by the same delta d.

        ~ An earlier version of this comment called the shift an exact
        reparametrisation with a BIT-IDENTICAL observation, via
        `(q + d) - (default + d)`.  THAT IS FALSE.  q is the MEASURED joint
        angle and carries no `+d`, so the policy's INPUT moves by
        `-d * dof_pos_scale` (= -d; dof_pos_scale is 1.0) while the readout
        gains `+d`.  The dual shift is still the right build, because the two
        terms OPPOSE and the policy is compliant -- it absorbs only part of
        the input perturbation, so the loop settles short of d rather than
        cancelling it.

        Measured in the pinned scene with the real Balance ONNX and the real
        PD (tools/mj_stance_probe.py, mode shift-both, deg=4): hip roll
        L +1.26 -> +3.44 deg, R -1.21 -> -3.35 deg, ankle separation
        +2.82 cm, and it HELD -- +0.13 cm of drift over the 3 s after the
        mechanism was removed.  Shift ONE side alone instead and the same
        probe's M5 row applies: a constant-torque chase with no fixed point.

        So the splayed stance is a state the policy HOLDS, not a standing
        error the PD pulls against.  (A hand-set splay survived a descend, a
        pick and a retreat at kp=150, which a restoring hip-roll loop could
        not have allowed.)

        Two properties the pins assert, both load-bearing:
          * IDEMPOTENT — always computed from the boot snapshot, so repeating a
            value changes nothing and `stance off` restores the boot encoding.
          * ATOMIC — whole arrays are REBOUND, never mutated in place, so the
            250 Hz tick can only ever read one consistent encoding and never a
            torn half-applied delta.
        """
        d2r = math.pi / 180.0
        hip = deg * d2r
        ank = self.stance_ankle_ratio * hip
        # 1=L hip, 5=L ankle, 7=R hip, 11=R ankle. Each ankle keeps
        # its hip's sign so the sole tilt holds flat; reversed signs
        # tilt the sole 1 deg per degree.
        d_pol = np.zeros_like(self._stance_base_policy)
        d_pol[1] = +hip
        d_pol[7] = -hip
        d_pol[5] = +ank
        d_pol[11] = -ank
        d_obs = np.zeros_like(self._stance_base_obs)
        d_obs[0:15] = d_pol       # 15 legs+waist
        self.policy.defaults = self._stance_base_policy + d_pol
        self.obs.default_padded = self._stance_base_obs + d_obs
        self.stance_deg = deg
        if deg == 0.0:
            self._stance_log("off -> boot stance", force=True)
        else:
            self._stance_log(f"{deg:+.1f} deg splay "
                             f"(ankle follows {self.stance_ankle_ratio:+.3f})",
                             force=True)

    def _stance_log(self, msg, force=False):
        """Rate-limited (1 Hz) stance log — no assert, no exception, matching
        the discipline every other lane in this file follows."""
        now = time.monotonic()
        if force or now - self._stance_log_t >= 1.0:
            self._stance_log_t = now
            print(f"[stance] {msg}")

    def _stance_status(self):
        """Inline status fragment, same shape as _cmd_auto_status."""
        return f"stance={self.stance_deg:+.1f}(max {self.stance_deg_max:.1f})"

    # ------------------------------------------------------- runtime step lane
    # same-sign hip-roll shift, lift and place the unloaded foot, then
    # closed-chain re-centre. Every constant is derived from the URDF
    # forward kinematics, not measured.
    # additive: own verb only, re-derived from boot, never compounded.
    # shares stance_ankle_ratio.

    def _init_step_state(self):
        """Step-lane state.  SAFE BY DEFAULT TWICE OVER: BOTH bands ship SHUT
        (`step_shift_deg_max` and `step_cm_max` default 0.0), and `step go`
        REFUSES while either is shut — a clamp alone would still lift a leg
        with no unload and no widening, because the clearance is not a band.
        Arming is a deliberate config edit on the box.

        Every key is read with `.get(..., <safe default>)` for the same reason
        _init_stance_state does — an older config must not raise KeyError at
        construction."""
        self.step_shift_deg_max = float(self.cfg.get("step_shift_deg_max", 0.0))
        if not np.isfinite(self.step_shift_deg_max) \
                or self.step_shift_deg_max < 0.0:
            self.step_shift_deg_max = 0.0     # NaN/neg = shut
        self.step_cm_max = float(self.cfg.get("step_cm_max", 0.0))
        if not np.isfinite(self.step_cm_max) or self.step_cm_max < 0.0:
            self.step_cm_max = 0.0            # NaN/neg = shut
        self.step_slew_deg_s = self._step_num("step_slew_deg_s", 8.0, 1e-6)
        self.step_lift_slew_deg_s = self._step_num("step_lift_slew_deg_s",
                                                   6.0, 1e-6)
        self.step_clearance_cm = self._step_num("step_clearance_cm", 3.0, 0.0)
        self.step_knee_max_deg = self._step_num("step_knee_max_deg", 20.0, 1e-6)
        self.step_min_shift_deg = self._step_num("step_min_shift_deg", 0.0, 0.0)
        self.step_unload_tau_x = self._step_num("step_unload_tau_x", 2.0, 0.0)
        self.step_hold_s = self._step_num("step_hold_s", 2.0, 0.0)
        self.step_tilt_deg = self._step_num("step_tilt_deg", 8.0, 1e-6)
        self.step_ankle_margin_deg = float(
            self.cfg.get("step_ankle_margin_deg", 2.0))
        if not np.isfinite(self.step_ankle_margin_deg):
            self.step_ankle_margin_deg = 2.0
        # R ankle has 14 deg of headroom; a larger margin flips the sign.
        self.step_ankle_margin_deg = min(
            max(self.step_ankle_margin_deg, 0.0),
            STEP_ANKLE_STOP_DEG - STEP_ANKLE_BASE_DEG)
        self.step_shift_deg = 0.0          # installed shift
        self.step_shift_target = 0.0       # commanded shift
        self.step_c1 = 0.0                 # L hip delta, deg
        self.step_c7 = 0.0                 # R hip delta, deg
        self.step_c1_t = 0.0
        self.step_c7_t = 0.0
        self.step_splay_l = 0.0            # outward splay, deg
        self.step_splay_r = 0.0
        self.step_splay_l_t = 0.0
        self.step_splay_r_t = 0.0
        self.step_k = 0.0                  # lift gain K
        self.step_k_t = 0.0
        self.step_phase = "IDLE"           # IDLE|SHIFT|HOLD|LIFT|PLACE|
                                           # RECENTRE|DONE|ABORT
        self.step_leg = ""                 # swing leg l|r
        self.step_cm_req = 0.0             # requested cm
        self.step_cm_ach = 0.0             # achieved cm
        self.step_w_m = 0.0                # widening, m
        self.step_tau0 = None              # unload baseline
        self.step_dtau = None
        self.step_hold_t = None            # gate first held
        self.step_go_pending = False
        self.step_abort_reason = ""
        self.step_tilt = None
        self.step_log_t = 0.0
        self._step_last_key = None         # last pose

    def _step_num(self, key, default, lo):
        """Config float with a safe fallback: missing, non-finite or below
        `lo` returns `default`.  A slew rate of 0 (or a negative one) would
        freeze the trajectory forever, so `lo` is 1e-6 for the rates."""
        try:
            v = float(self.cfg.get(key, default))
        except (TypeError, ValueError):
            return default
        if not np.isfinite(v) or v < lo:
            return default
        return v

    def _step_write(self):
        """Install the current pose into both encodings — but ONLY when the
        pose key changed.  This is the additive guarantee's second half: an
        IDLE lane never reaches this call, and an arrived slew calls it once.
        `_write_step` REBINDS both arrays from the boot snapshot, so a write is
        atomic and can never compound (the A5/A6/A15 properties)."""
        key = (round(self.step_c1, 12), round(self.step_c7, 12),
               round(self.step_k, 12), self.step_leg)
        if key == self._step_last_key:
            return
        self._step_last_key = key
        self._write_step(self.step_c1, self.step_c7, self.step_k,
                         self.step_leg if self.step_k > 0.0 else None)

    def _write_step(self, d1_deg, d7_deg, k_deg=0.0, leg=None):
        """Install the step offset into BOTH policy-side encodings.

        THE MECHANISM, as MEASURED (the stance lane's probe, mj_stance_probe.py
        mode shift-both): both encodings are re-derived from the BOOT snapshot
        and shifted by the same delta d.  policy_runner:50 reads
        `target_q = action*action_scale + defaults`; obs_builder:45-46 builds
        `default_padded` (15 defaults + 14 zeros) and `f[13:42] = q -
        default_padded`.  The policy INPUT therefore moves by -d while the
        readout gains +d; the terms OPPOSE and the compliant policy absorbs
        part of it (86% at 4 deg), so a COMMANDED shift is not an ACHIEVED one
        and every log line prints both columns.

        Sign law (§2.2, asymmetric — do not "fix" it):
          d_pol[1] = +d1   L hip roll; +(idx 1) abducts the L leg
          d_pol[7] = +d7   R hip roll; for the R leg it is -(idx 7) that
                           abducts, so a SAME-SIGN pair +x translates the
                           pelvis toward the robot's RIGHT and unloads the L
                           foot.  `step shift +x` therefore always steps the L
                           foot first.
          d_pol[5] = ratio*d1, d_pol[11] = ratio*d7 — the flat-sole co-motion
          (one source of truth: `stance_ankle_ratio`).  Getting one of these
          backwards drives the sole tilt at ~1.0 deg per degree.

        The lift triple (leg "l" -> 0/3/4, "r" -> 6/9/10) is the unique plane
        law that raises the foot while holding its fore-aft position and its
        sole attitude: d_hip_pitch = -0.472K, d_knee = +K, d_ankle_pitch =
        -0.528K, dz = 0.2333 cm per degree of K.  DERIVED (sim + repo FK).
        Waist (12-14) is never written; the arms (15-28) are NOT in this
        channel — d_obs[15:] stays zero (mutation M9 pins that slot)."""
        d2r = math.pi / 180.0
        d_pol = np.zeros_like(self._stance_base_policy)
        d_pol[1] = d1_deg * d2r
        d_pol[7] = d7_deg * d2r
        d_pol[5] = self.stance_ankle_ratio * d_pol[1]
        d_pol[11] = self.stance_ankle_ratio * d_pol[7]
        k = k_deg * d2r
        if leg == "l":
            d_pol[0] = STEP_LIFT_HIP_PITCH * k
            d_pol[3] = k
            d_pol[4] = STEP_LIFT_ANKLE_PITCH * k
        elif leg == "r":
            d_pol[6] = STEP_LIFT_HIP_PITCH * k
            d_pol[9] = k
            d_pol[10] = STEP_LIFT_ANKLE_PITCH * k
        d_obs = np.zeros_like(self._stance_base_obs)
        d_obs[0:15] = d_pol       # 15 legs+waist
        self.policy.defaults = self._stance_base_policy + d_pol
        self.obs.default_padded = self._stance_base_obs + d_obs
        self.step_c1 = d1_deg
        self.step_c7 = d7_deg
        self.step_k = k_deg

    def _step_cap_l(self):
        """Max L hip-roll delta, deg — DERIVED: the L ankle's -15 deg stop with
        the boot's +1.0 deg of headroom, (15 + 1 - margin)/|ratio| = +16.47 at
        the shipped margin.  NOT interchangeable with the R cap (P4)."""
        r = max(abs(self.stance_ankle_ratio), 1e-9)
        return (STEP_ANKLE_STOP_DEG + STEP_ANKLE_BASE_DEG
                - self.step_ankle_margin_deg) / r

    def _step_cap_r(self):
        """Min R hip-roll delta, deg — DERIVED: the R ankle's +15 deg stop,
        which has only 14 deg of headroom because the boot base is +1.0 on
        BOTH legs, -(15 - 1 - margin)/|ratio| = -14.12 at the shipped margin.
        The L/R asymmetry is REAL and derived; symmetrising it is the bug."""
        r = max(abs(self.stance_ankle_ratio), 1e-9)
        return -(STEP_ANKLE_STOP_DEG - STEP_ANKLE_BASE_DEG
                 - self.step_ankle_margin_deg) / r

    def _step_shift_cap(self):
        """Max POSITIVE same-sign shift, deg: a positive shift drives BOTH
        ankles toward their -15 deg stops and both have 16 deg of headroom from
        the +1.0 deg boot base — the same number as _step_cap_l."""
        return self._step_cap_l()

    def _step_shift_limits(self):
        """(lo, hi) for the shift, from the ankle stops.  With margin 0 these
        are the spec's hard limits -16.47 / +18.82 deg — asymmetric, because
        the boot base is +1.0 on both legs and the two directions spend it
        differently."""
        return self._step_cap_r(), self._step_shift_cap()

    def _step_meas(self, st):
        """Measured joint columns, degrees, or None when the state lacks them
        (fail-closed: no measurement, no motion).  `st["q"]` is RADIANS."""
        if st is None:
            return None
        try:
            q = st["q"]
            r2d = 180.0 / math.pi
            return {"hip1": float(q[1]) * r2d, "hip7": float(q[7]) * r2d,
                    "ank5": float(q[5]) * r2d, "ank11": float(q[11]) * r2d,
                    "knee3": float(q[3]) * r2d, "knee9": float(q[9]) * r2d}
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _step_taus(self, st):
        """(tau[3], tau[9]) — the knee torques the unload gate reads
        (state_reader.py:108 `"tau": tau,`) — or (None, None)."""
        if st is None:
            return None, None
        try:
            tau = st["tau"]
            return float(tau[3]), float(tau[9])
        except (KeyError, IndexError, TypeError, ValueError):
            return None, None

    def _step_curve_c7(self, d1_deg, w_m):
        """The closed-chain curve (§2.3 P5): with both feet planted the
        widening W = 0.4955*(sin d1 - sin d7) is FIXED, so d7 is a function of
        d1.  Only poses ON this curve keep both feet planted — any other pose
        demands a foot SLIDE, which §3.1 proves is impossible (~20 N available
        vs ~85-170 N needed).  DERIVED, and the most important safety rule in
        the lane."""
        s = math.sin(d1_deg * math.pi / 180.0) - w_m / STEP_SEP_LEVER_M
        return math.degrees(math.asin(min(max(s, -1.0), 1.0)))

    def _step_symmetric_d1(self, w_m):
        """The d1 of the SYMMETRIC end pose (d1 = +h, d7 = -h), h =
        asin(W/0.991): the CoM returns to the middle of the two feet.  This is
        where the re-centre walks to, and it is a stable state, not a transient."""
        h = w_m / (2.0 * STEP_SEP_LEVER_M)
        return math.degrees(math.asin(min(max(h, -1.0), 1.0)))

    def _step_target_c1(self, w_m, c_deg):
        """The d1 of the pose on the closed-chain curve whose COMMON MODE
        (d1 + d7)/2 equals c_deg — the lane's canonical target for a shift.
        With W == 0 the curve is d1 == d7, so the answer is exact; with W > 0
        it is found by bisection (d7 rises with d1, so the relation is
        monotone).  DERIVED."""
        if abs(w_m) < 1e-12:
            return c_deg
        lo, hi = c_deg - 60.0, c_deg + 60.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if 0.5 * (mid + self._step_curve_c7(mid, w_m)) < c_deg:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _step_approach(self, cur, tgt, rate, dt):
        """One slew step toward tgt; lands EXACTLY on tgt, which is what the
        settled tests compare against.  Never raises, never asserts."""
        d = tgt - cur
        step = rate * dt
        if abs(d) <= step:
            return tgt
        return cur + (step if d > 0.0 else -step)

    def _step_r_budget(self):
        """The §2.3 P6 derived number: how much OUTWARD splay the R leg can
        still take after paying for the NEGATIVE shift that unloads it —
        (15 - 1 - margin)/|ratio| - |shift|.  For a shift >= 0 the R leg is not
        the swing leg and the number is 0.0 (the value the spec's own example
        prints).  A budget <= 0 means: at the R ankle's stop there is no room
        left to widen, so `step go r` is refused.  DERIVED, never measured."""
        if self.step_c7_t >= 0.0:
            return 0.0
        return (-self._step_cap_r()) - abs(self.step_c7_t)

    def _step_splay_target(self, leg, cm_m):
        """Map a commanded widening <cm> to a hip-roll splay, then clamp it by
        the PLANT caps (P4) — which are asymmetric for L and R and must NOT be
        symmetrised.  Returns (clamped splay, raw splay, cap), degrees.

        Lever law: the widening is 0.4955*(sin d1 - sin d7) m, so the swing
        leg's target hip roll is asin(sin(held) +/- cm/0.4955) and the splay is
        its difference from the held value.  (`held` is the TARGET of the
        leg that stays planted, so a `go` issued before the shift has settled
        is still planned against the settled shift.)  The spec's log template
        also quotes this law as 86.5 cm; the two disagree by 1.746x and this
        build uses the self-consistent 0.4955 m lever — see _step_detail."""
        if leg == "l":
            held = self.step_c7_t
            want = math.degrees(math.asin(min(max(
                math.sin(math.radians(held)) + cm_m / STEP_SEP_LEVER_M,
                -1.0), 1.0)))
            raw = want - held
            cap = self._step_cap_l() - self.step_c1_t
        else:
            held = self.step_c1_t
            want = math.degrees(math.asin(min(max(
                math.sin(math.radians(held)) - cm_m / STEP_SEP_LEVER_M,
                -1.0), 1.0)))
            raw = held - want
            cap = self.step_c7_t - self._step_cap_r()
        return max(0.0, min(raw, cap)), raw, cap

    def _apply_step(self, parts):
        """step parse+apply, shared by socket and console (the same lockstep
        split that keeps stance/height/cmd_set identical at both sites).
        Returns an error message or None on success:
          step shift <deg>    same-sign hip-roll weight shift; clamped to the
                              operator band, SHIPPED SHUT (band 0.0)
          step go <l|r> <cm>  lift + place that foot; DONE = STARTED
                              (completion is polled via `step status`)
          step off            abort anything in flight, restore the BOOT
                              encoding bit-exactly (the `stance off` property)
          step status         no-op here; the caller prints the fragment

        Every refusal is FAIL-CLOSED and names its reason (§6.2): a refusal
        never writes an array, so the measured column does not move."""
        if len(parts) >= 2 and parts[1] == "status":
            return None
        if self._stance_base_policy is None:
            return "step unavailable: no policy (stage 0)"
        if len(parts) == 2 and parts[1] == "off":
            self._step_reset(force=True)
            return None
        if len(parts) == 3 and parts[1] == "shift":
            try:
                deg = float(parts[2])
            except ValueError:
                return "step needs a float"
            if not np.isfinite(deg):
                return "step value must be finite"
            if self.stance_deg != 0.0:
                return "step shift needs stance idle"
            if self.step_go_pending or self.step_phase not in (
                    "IDLE", "DONE", "SHIFT", "ABORT"):
                return "step busy: %s" % self.step_phase
            deg = min(max(deg, -self.step_shift_deg_max),
                      self.step_shift_deg_max)
            lo, hi = self._step_shift_limits()
            deg = min(max(deg, lo), hi)   # ankle stop clamp
            if deg != self.step_shift_target:
                self.step_tau0 = None     # fresh baseline
            self.step_shift_target = deg
            self.step_c1_t = self._step_target_c1(self.step_w_m, deg)
            self.step_c7_t = self._step_curve_c7(self.step_c1_t, self.step_w_m)
            self.step_leg = ""
            self.step_hold_t = None
            self.step_phase = "SHIFT"
            self._step_write()
            return None
        if len(parts) == 4 and parts[1] == "go":
            leg = parts[2].lower()
            try:
                cm = float(parts[3])
            except ValueError:
                return "step needs a float"
            if not np.isfinite(cm):
                return "step value must be finite"
            if leg not in ("l", "r"):
                return "step go needs l|r"
            if self.step_go_pending or self.step_phase not in ("IDLE", "DONE"):
                return "step busy: %s" % self.step_phase
            if self.payload:
                return "step go refuses with payload set"
            if self.step_shift_deg_max <= 0.0 or self.step_cm_max <= 0.0:
                return ("step go refuses: band shut (step_shift_deg_max=%.1f "
                        "step_cm_max=%.1f) — arm both in the live config"
                        % (self.step_shift_deg_max, self.step_cm_max))
            sgn = 1.0 if leg == "l" else -1.0
            if sgn * self.step_shift_target < self.step_min_shift_deg:
                return ("step go needs an installed shift >= %.1f deg "
                        "(have %+.1f for leg %s)"
                        % (self.step_min_shift_deg, self.step_shift_target,
                           leg))
            self.step_leg = leg
            self.step_cm_req = min(max(cm, 0.0), self.step_cm_max)
            tgt, raw, cap = self._step_splay_target(leg,
                                                    self.step_cm_req * 0.01)
            self._step_go_log()
            if cap <= 0.0:
                self.step_leg = ""
                return ("step go %s refuses: no outward ankle budget left "
                        "(commanded %.1f deg splay, cap %.2f deg); hold at "
                        "the current separation" % (leg, raw, cap))
            if leg == "l":
                self.step_splay_l_t = tgt
                self.step_splay_r_t = 0.0
            else:
                self.step_splay_r_t = tgt
                self.step_splay_l_t = 0.0
            self.step_k_t = min(self.step_clearance_cm
                                / STEP_LIFT_DZ_CM_PER_DEG,
                                self.step_knee_max_deg)
            self.step_tau0 = None       # fresh baseline
            self.step_hold_t = None
            self.step_go_pending = True
            self.step_phase = "SHIFT"
            self._step_write()
            return None
        return "step off | step shift <deg> | step go l|r <cm> | step status"

    def _step_reset(self, force=False):
        """Return the lane to IDLE and the encodings to BOOT.  Used by
        `step off`, by the tilt abort and by shutdown.  The relax is a SNAP,
        exactly like `stance off` (`_write_stance(0.0)`), so the two verbs
        behave alike and the P7 property (array_equal(defaults, boot)) holds
        bit-exactly.  `force` prints the P7 line; a shutdown must not.

        PHYSICS NOTE: after a COMPLETED step the feet are wider on the floor,
        so a snap to boot asks the legs to close a separation the feet cannot
        slide; the policy holds what it can and the live run measures it.  The
        snap is what the spec pins (A15), not a measured claim."""
        self.step_phase = "IDLE"
        self.step_leg = ""
        self.step_shift_deg = 0.0
        self.step_shift_target = 0.0
        self.step_c1 = 0.0
        self.step_c7 = 0.0
        self.step_c1_t = 0.0
        self.step_c7_t = 0.0
        self.step_splay_l = 0.0
        self.step_splay_r = 0.0
        self.step_splay_l_t = 0.0
        self.step_splay_r_t = 0.0
        self.step_k = 0.0
        self.step_k_t = 0.0
        self.step_w_m = 0.0
        self.step_cm_req = 0.0
        self.step_cm_ach = 0.0
        self.step_go_pending = False
        self.step_tau0 = None
        self.step_dtau = None
        self.step_hold_t = None
        self.step_tilt = None
        self._step_write()
        if force:
            self._step_log("off -> boot encoding restored (bit-exact)",
                           force=True)

    def _abort_step(self, reason):
        """Self-abort (a tilt, or a shutdown while in flight): the trajectory
        stops, the encoding returns to BOOT, and the mandated line prints the
        phase and shift it stopped in."""
        phase = self.step_phase
        shift = self.step_shift_deg
        self._step_reset(force=False)
        self.step_phase = "ABORT"
        self.step_abort_reason = reason
        self._step_log("ABORT %s phase=%s shift=%+.1f — relaxing to boot "
                       "encoding" % (reason, phase, shift), force=True)

    def _step_gate_ok(self):
        """The §7 unload predicate: |dtau| >= step_unload_tau_x * |dtau0|.
        A DEGENERATE baseline (|dtau0| == 0, or a zero bar) CLOSES the gate:
        with bar == 0 the comparison is vacuously true, and a gate that cannot
        hold is not a gate.  Fail-closed — the trajectory holds and logs."""
        if self.step_tau0 is None or self.step_dtau is None:
            return False
        bar = self.step_unload_tau_x * abs(self.step_tau0)
        if bar <= 0.0:
            return False
        return abs(self.step_dtau) >= bar

    def _step_n_swing(self, tau_l, tau_r):
        """|tau_swing|/0.109 as an upper bound on the swing foot's load (§4);
        nan when the leg or the taus are unknown — a 0.0 would read as a
        perfect unload."""
        nan = float("nan")
        n_l = abs(tau_l) / STEP_KNEE_LEVER_M if tau_l is not None else nan
        n_r = abs(tau_r) / STEP_KNEE_LEVER_M if tau_r is not None else nan
        if self.step_leg == "l":
            return n_l
        if self.step_leg == "r":
            return n_r
        return min(n_l, n_r)

    def _step_log(self, msg, force=False):
        """Rate-limited (1 Hz) step log.  Every [step] body-state line goes
        through here so the honesty suffix is appended in EXACTLY one place,
        and exactly once per line (A12).  No assert, no exception."""
        now = time.monotonic()
        if force or now - self.step_log_t >= 1.0:
            self.step_log_t = now
            if not msg.endswith(STEP_HONESTY_SUFFIX):
                msg = "%s %s" % (msg, STEP_HONESTY_SUFFIX)
            print("[step] %s" % msg)

    def _step_hold_log(self, tau_l, tau_r):
        """The §6.3 signature: the shift is commanded and the gate is NOT met,
        so the lift is NOT licensed.  Rate-limited by _step_log."""
        if self.step_tau0 is None or self.step_dtau is None \
                or self.step_tau0 == 0.0:
            return
        bar = self.step_unload_tau_x * abs(self.step_tau0)
        self._step_log(
            "HOLD unload: dtau=%+.3f bar=%.3f (%.2fx) N_swing<=%.0fN — "
            "lifting is NOT licensed"
            % (self.step_dtau, bar, abs(self.step_dtau) / abs(self.step_tau0),
               self._step_n_swing(tau_l, tau_r)))

    def _step_go_log(self):
        """The §2.3 P6 / §4 derived lines: the mechanism's ceiling, the
        requested widening, the R leg's outward budget and the prediction that
        the balance policy absorbs part of every command (the sim's own
        'stance 4' settled at 86% of command).  Fires on every `step go`,
        including a refused one, so the refusal carries its number."""
        ank = float("nan")
        st = getattr(self, "lowstate", None)
        m = self._step_meas(st) if st is not None else None
        if m is not None:
            ank = m["ank11"]
        self._step_log(
            "derived cap: this mechanism's max WIDENING ~%.1f cm on top of the "
            "boot stance — i.e. a total ankle-to-ankle separation of ~52.2 cm "
            "(sim, pinned model §2.1), NOT a %.1f cm total; requested %.1f cm "
            "of widening; the balance policy may "
            "absorb part of every command (sim 'stance 4' settled at 86%% of "
            "command)." % (STEP_TWO_STEP_MAX_CM, STEP_TWO_STEP_MAX_CM,
                           self.step_cm_req), force=True)
        self._step_log("cap: R-leg outward budget %.2f deg (ankle %+.2f of 15)"
                       % (self._step_r_budget(), ank), force=True)

    def _step_detail(self, st, force=False):
        """The §4 instrumentation line — ONE line, rate-limited 1 Hz, forced on
        every phase transition.  Commanded AND measured columns both appear:
        the sim's own evidence is that the loop settles short of the command
        (86% at 4 deg), so only the measured column is evidence.  Unavailable
        measurements print as nan, never 0.0 — a default would make a missing
        state look perfectly centred.

        DERIVED fields (§4): dtau = tau[3] - tau[9]; N = |tau|/0.109; the CoM
        travel is 0.6856 cm per degree of same-sign shift; pred_sep_gain is the
        separation gain implied by the MEASURED hip rolls.  NOTE: §4's template
        quotes that last law as `86.5*(sin d1 - sin d7)` cm, which disagrees
        with the 0.4955 m lever used everywhere else in the spec (§0, §2.3
        P4's own parenthetical, P5 and A9) by a factor of 1.746; this build
        uses the self-consistent lever (100*0.4955 = 49.55), which is the only
        value that reproduces the spec's own worked examples (+2.7 cm at a
        1 deg margin, Annex A #9).  Flagged to the operator."""
        now = time.monotonic()
        if not force and now - self.step_log_t < 1.0:
            return
        self.step_log_t = now
        nan = float("nan")
        m = self._step_meas(st) or {}
        h1 = m.get("hip1", nan)
        h7 = m.get("hip7", nan)
        a5 = m.get("ank5", nan)
        a11 = m.get("ank11", nan)
        tau_l, tau_r = self._step_taus(st)
        n_l = abs(tau_l) / STEP_KNEE_LEVER_M if tau_l is not None else nan
        n_r = abs(tau_r) / STEP_KNEE_LEVER_M if tau_r is not None else nan
        base = self._stance_base_policy
        if base is not None:
            b1 = math.degrees(float(base[1]))
            b7 = math.degrees(float(base[7]))
            cmd1 = b1 + self.step_c1 + self.step_splay_l
            cmd7 = b7 + self.step_c7 - self.step_splay_r
            splay_l_meas = h1 - b1 - self.step_c1
            splay_r_meas = b7 + self.step_c7 - h7
        else:
            cmd1 = cmd7 = splay_l_meas = splay_r_meas = nan
        dtau = self.step_dtau if self.step_dtau is not None else nan
        tau0 = self.step_tau0 if self.step_tau0 is not None else nan
        bar = (self.step_unload_tau_x * abs(self.step_tau0)
               if self.step_tau0 is not None else nan)
        ratio = (abs(self.step_dtau) / abs(self.step_tau0)
                 if (self.step_dtau is not None and self.step_tau0) else nan)
        tilt = self.step_tilt if self.step_tilt is not None else nan
        gain = 100.0 * STEP_SEP_LEVER_M * (
            math.sin(math.radians(h1)) - math.sin(math.radians(h7)))
        self._step_log(
            "%s shift cmd=%+.1f meas_hip=%+.2f/%+.2f(dL=%+.2f) | "
            "dtau=%+.3f tau0=%+.3f x=%.2f bar=%.3f | "
            "tauL=%+.3f tauR=%+.3f N_L=%.0fN N_R=%.0fN N_swing<=%.1fN | "
            "tilt=%.1f | "
            "splay cmd=%.1f/%.1f meas=%.1f/%.1f ankle=%+.2f/%+.2f "
            "margin=%.2f/%.2f | "
            "cap: shift<=%.2f L<=%.2f R>=%.2f pred_sep_gain=%.2fcm"
            % (self.step_phase, self.step_shift_deg, h1, h7,
               h1 - cmd1 if base is not None else nan,
               dtau, tau0, ratio, bar,
               tau_l if tau_l is not None else nan,
               tau_r if tau_r is not None else nan,
               n_l, n_r, self._step_n_swing(tau_l, tau_r), tilt,
               self.step_splay_l_t, self.step_splay_r_t,
               splay_l_meas, splay_r_meas, a5, a11,
               STEP_ANKLE_STOP_DEG - abs(a5), STEP_ANKLE_STOP_DEG - abs(a11),
               self._step_shift_cap(), self._step_cap_l(), self._step_cap_r(),
               gain),
            force=True)

    def _step_status(self):
        """Inline status fragment, same shape as _stance_status.  The terminal
        text names the OUTCOME (§1.4: `step go` returns DONE = STARTED and
        completion is polled here).  Ends with the honesty suffix exactly once
        — the fragment reports a body state (a phase and a shift), so §3 says
        the suffix is not optional."""
        ph = self.step_phase
        nan = float("nan")
        if ph == "IDLE":
            frag = "step=IDLE shift=%+.1f" % self.step_shift_deg
        elif ph == "SHIFT":
            frag = "step=SHIFT(%.1f)" % self.step_shift_target
        elif ph == "HOLD":
            frag = "step=HOLD(unload dtau=%+.1f bar %.1f)" % (
                self.step_dtau if self.step_dtau is not None else nan,
                (self.step_unload_tau_x * abs(self.step_tau0)
                 if self.step_tau0 is not None else nan))
        elif ph == "LIFT":
            frag = "step=LIFT(%s K=%.1f)" % (self.step_leg, self.step_k_t)
        elif ph in ("PLACE", "RECENTRE"):
            frag = "step=%s(%s %.2fcm)" % (ph, self.step_leg,
                                           self.step_cm_req)
        elif ph == "DONE":
            frag = "step=DONE(%s %.2fcm shift=%+.1f)" % (
                self.step_leg, self.step_cm_ach, self.step_shift_deg)
        elif ph == "ABORT":
            frag = "step=ABORT(%s)" % self.step_abort_reason
        else:
            frag = "step=%s" % ph
        return "%s %s" % (frag, STEP_HONESTY_SUFFIX)

    def _step_advance(self, t, st):
        """The step lane's tick hook (§1.3h) — called every 250 Hz tick from
        `_tick`, immediately BEFORE the policy tick, so the encoding it writes
        is the one the policy consumes in the SAME frame.

        ADDITIVE: the first line returns unless a step verb has armed the lane
        (phase IDLE/ABORT) or the boot snapshot is absent, so with the lane
        untouched the byte path is the pre-step writer's.  Every exit is
        fail-closed: no state, no motion, no exception."""
        if self._stance_base_policy is None:
            return
        ph0 = self.step_phase
        if ph0 in ("IDLE", "ABORT"):
            return
        meas = self._step_meas(st)
        if meas is None:
            return
        # tilt abort: self-aborts to boot, never waits.
        self.step_tilt = None
        if SC is not None:
            try:
                self.step_tilt = float(SC.tilt_deg(st["quat"]))
            except Exception:
                self.step_tilt = None
        if self.step_tilt is not None and self.step_tilt > self.step_tilt_deg:
            self._abort_step("tilt %.1f" % self.step_tilt)
            return
        tau_l, tau_r = self._step_taus(st)
        self.step_dtau = (tau_l - tau_r
                          if (tau_l is not None and tau_r is not None)
                          else None)
        # baseline = the first tick, before any slew, so the unload gate
        # compares against this run's own loading.
        if self.step_tau0 is None:
            if tau_l is not None and tau_r is not None:
                self.step_tau0 = tau_l - tau_r
                self.step_hold_t = None
                self._step_log(
                    "tau0 captured at P0: tauL=%+.3f tauR=%+.3f dtau0=%+.3f "
                    "(the run's own baseline — the doc's 1.32 was under "
                    "stance 4)" % (tau_l, tau_r, self.step_tau0), force=True)
        dt = self.dt
        if self.step_phase in ("SHIFT", "HOLD"):
            # slew the shift in, hold until unload gate opens.
            if self.step_w_m == 0.0:
                self.step_c1 = self._step_approach(
                    self.step_c1, self.step_c1_t, self.step_slew_deg_s, dt)
                self.step_c7 = self._step_approach(
                    self.step_c7, self.step_c7_t, self.step_slew_deg_s, dt)
                settled = (self.step_c1 == self.step_c1_t
                           and self.step_c7 == self.step_c7_t)
            else:
                self.step_c1 = self._step_approach(
                    self.step_c1, self.step_c1_t, self.step_slew_deg_s, dt)
                self.step_c7 = self._step_curve_c7(self.step_c1, self.step_w_m)
                settled = (self.step_c1 == self.step_c1_t)
            self.step_shift_deg = 0.5 * (self.step_c1 + self.step_c7)
            if not self.step_go_pending:
                if settled:
                    self.step_phase = "IDLE"
            else:
                if settled:
                    self.step_phase = "HOLD"
                if self._step_gate_ok():
                    if self.step_hold_t is None:
                        self.step_hold_t = t
                    if settled and (t - self.step_hold_t) >= self.step_hold_s:
                        self.step_phase = "LIFT"
                        self.step_k_t = min(
                            self.step_clearance_cm / STEP_LIFT_DZ_CM_PER_DEG,
                            self.step_knee_max_deg)
                        self._step_log(
                            "unload PROVEN: dtau=%+.3f (%.2fx)"
                            % (self.step_dtau,
                               abs(self.step_dtau) / abs(self.step_tau0)),
                            force=True)
                else:
                    self.step_hold_t = None
                    self._step_hold_log(tau_l, tau_r)
        elif self.step_phase == "LIFT":
            # lift the swing foot on the plane law.
            # stance leg holds the shift
            self.step_k = self._step_approach(
                self.step_k, self.step_k_t, self.step_lift_slew_deg_s, dt)
            self.step_splay_l = self._step_approach(
                self.step_splay_l, self.step_splay_l_t, self.step_slew_deg_s,
                dt)
            self.step_splay_r = self._step_approach(
                self.step_splay_r, self.step_splay_r_t, self.step_slew_deg_s,
                dt)
            if (self.step_k == self.step_k_t
                    and self.step_splay_l == self.step_splay_l_t
                    and self.step_splay_r == self.step_splay_r_t):
                self.step_phase = "PLACE"
                self.step_k_t = 0.0
        elif self.step_phase == "PLACE":
            # lower the triple; plant caps already clamped it flat.
            self.step_k = self._step_approach(self.step_k, 0.0,
                                              self.step_lift_slew_deg_s, dt)
            self.step_splay_l = self._step_approach(
                self.step_splay_l, self.step_splay_l_t, self.step_slew_deg_s,
                dt)
            self.step_splay_r = self._step_approach(
                self.step_splay_r, self.step_splay_r_t, self.step_slew_deg_s,
                dt)
            if (self.step_k == 0.0
                    and self.step_splay_l == self.step_splay_l_t
                    and self.step_splay_r == self.step_splay_r_t):
                # planted: fold the splay into (c1, c7) and lock in the
                # widening, which then constrains every later pose.
                self.step_c1 = self.step_c1 + self.step_splay_l
                self.step_c7 = self.step_c7 - self.step_splay_r
                self.step_splay_l = 0.0
                self.step_splay_r = 0.0
                self.step_w_m = STEP_SEP_LEVER_M * (
                    math.sin(math.radians(self.step_c1))
                    - math.sin(math.radians(self.step_c7)))
                self.step_cm_ach = 100.0 * self.step_w_m
                self.step_c1_t = self._step_symmetric_d1(self.step_w_m)
                self.step_c7_t = -self.step_c1_t
                self.step_phase = "RECENTRE"
        elif self.step_phase == "RECENTRE":
            # walk to the symmetric end pose (d1=+h, d7=-h).
            # CoM centred; every pose on the curve, no sliding.
            self.step_c1 = self._step_approach(
                self.step_c1, self.step_c1_t, self.step_slew_deg_s, dt)
            self.step_c7 = self._step_curve_c7(self.step_c1, self.step_w_m)
            self.step_shift_deg = 0.5 * (self.step_c1 + self.step_c7)
            if self.step_c1 == self.step_c1_t:
                self.step_c7_t = self.step_c7
                self.step_cm_ach = 100.0 * STEP_SEP_LEVER_M * (
                    math.sin(math.radians(self.step_c1))
                    - math.sin(math.radians(self.step_c7)))
                self.step_phase = "DONE"
        self._step_write()
        self._step_detail(st, force=(self.step_phase != ph0))

    def _legs_ascended(self, st):
        """True when the measured leg joints (motors 0-11) are within
        height_ascend_tol of the POLICY-MEASURED standing pose — a base-height
        proxy that needs no FK.  shutdown() holds the policy live until this is
        True (bounded by height_ascend_timeout_s) before the limp frame.

        The reference is `height_stand_q`, built from config
        `policy_stand_q_legs` (the measured fixed point, see _init_height_state)
        — NOT the YAML default_angles, which the real policy misses by up to
        0.44 rad (FINDING 2: the old predicate was unsatisfiable at standing)."""
        if st is None:
            return False
        q = st["q"][0:12]
        return bool(np.all(np.abs(q - self.height_stand_q[0:12])
                           <= self.height_ascend_tol))

    # --------------------------------------------------- additive SONIC lane
    # additive: lane off and b == 0 is bit-identical to the GR00T-only
    # path: same ndarray, same gains, same flow.
    # no asserts; config reads use cfg.get so sparse configs degrade.

    def _init_sonic_state(self):
        """SONIC-lane state + tunables (spec §3, §6).  FORCES the lane disabled
        when sonic_core failed to import: the lane must degrade to "absent",
        never to "half-present"."""
        self.sonic_lane_enabled = bool(self.cfg.get("sonic_lane_enabled", False))
        if SC is None:
            self.sonic_lane_enabled = False    # no tables, no lane
        # state vars
        self.sonic_mode = 'off'
        self.sonic_armed = False
        self.sonic_b = 0.0
        self.sonic_target = None
        self.sonic_last_rx = 0.0
        self.sonic_rx = 0
        self.sonic_rej = 0
        self.sonic_seq = -1
        self.sonic_warm = 0
        self.sonic_last_good_t = 0.0
        self.sonic_faulted = False
        self.sonic_fault_cause = "none"
        self.sonic_prev_blend = 0.0
        self.sonic_has_frame = False
        self.sonic_log_t = 0.0
        self.sonic_overrun_head = 0
        self.sonic_overrun_base = 0
        self.sonic_kp_boot = None     # filled by fill_gains()
        self.sonic_kd_boot = None
        # revoke trigger needs a latched mode_machine.
        self.sonic_mm_latch = None
        # tunables, read through cfg.get with the defaults here.
        self.sonic_stale_ms = float(self.cfg.get("sonic_stale_ms", 150))
        self.sonic_entry_ramp_s = float(self.cfg.get("sonic_entry_ramp_s", 1.0))
        self.sonic_handback_ramp_s = float(
            self.cfg.get("sonic_handback_ramp_s", 0.5))
        self.sonic_blend_max_step = float(
            self.cfg.get("sonic_blend_max_step_rad", 0.10))
        self.sonic_warmup_frames = int(self.cfg.get("sonic_warmup_frames", 100))
        # stale_ms alone bounds the warmup window.
        self.sonic_enable_tilt_deg = float(
            self.cfg.get("sonic_enable_tilt_deg", 15.0))
        self.sonic_tilt_fail_deg = float(
            self.cfg.get("sonic_tilt_fail_deg", 22.0))
        self.sonic_enable_tol_rad = float(
            self.cfg.get("sonic_enable_tol_rad", 0.35))
        self.sonic_rc_override_thresh = float(
            self.cfg.get("sonic_rc_override_thresh", 0.1))
        self.sonic_frame_step_max = float(
            self.cfg.get("sonic_frame_step_max_rad", 0.35))
        self.sonic_target_abs_max = float(
            self.cfg.get("sonic_target_abs_max_rad", 3.0))
        self.sonic_arm_mode = str(self.cfg.get("sonic_arm_mode", "park"))
        self.sonic_max_overruns_per_s = float(
            self.cfg.get("sonic_max_overruns_per_s", 5))
        # gate probe: opt-in, '' = off, so no file, no dir, no reply change.
        # forensics only: reads state and appends JSONL, never writes a
        # gain, target or mode. Runs on the console thread, not the rt loop.
        self.sonic_gate_probe_dir = str(
            self.cfg.get("sonic_gate_probe_dir", ""))
        self.sonic_gate_probe_max = int(
            self.cfg.get("sonic_gate_probe_max", 512))
        self.sonic_gate_probe_fh = None      # lazy open
        self.sonic_gate_probe_path = None
        self.sonic_gate_probe_n = 0
        self.sonic_gate_dev = None
        if self.sonic_lane_enabled:
            self._sonic_log("lane enabled (mode=off, disarmed)")
        else:
            print(f"[sonic] lane disabled (cfg sonic_lane_enabled="
                  f"{self.cfg.get('sonic_lane_enabled', False)}, "
                  f"sonic_core={'ok' if SC is not None else 'MISSING'})",
                  flush=True)

    def _apply_sonic(self, parts, from_console):
        """SONIC verb dispatch, shared by socket and console (the same lockstep
        split as cmd_set/height).  Returns the reply line, never raises.

        Arming is a HUMAN action: `sonic arm`, `sonic on` and `sonic reset` are
        console-only and are refused on the socket (spec §2.2).  The satellite
        can never arm itself, and no network/agent client can either.
        """
        if not self.sonic_lane_enabled:
            return "ERROR:sonic lane disabled"
        if len(parts) == 1:
            return f"SONIC:status {self._sonic_status()}"
        sub = parts[1]
        if sub == "status":
            return f"SONIC:status {self._sonic_status()}"
        if sub == "arm":
            if not from_console:
                return "ERROR:sonic arm is console-only"
            # arming does not clear a fault; the path is reset -> arm -> on.
            self.sonic_armed = True
            self._sonic_log(
                "ARMED (console) — fault latched, `sonic reset` first"
                if self.sonic_faulted else "ARMED (console)", force=True)
            return "SONIC:armed"
        if sub == "on":
            if not from_console:
                return "ERROR:sonic on is console-only"
            # going live: all of these must hold
            if not self.sonic_armed:
                return "ERROR:sonic not armed"
            if self.sonic_faulted:
                return "ERROR:sonic faulted — sonic reset first"
            if not self.sonic_has_frame:
                return "ERROR:sonic no frame received"
            # capture the enable-gate operand max|sonic_target - leg_target|
            # once, here, above the warmup check: the frame-step gate can
            # stall sonic_warm at 0/1, so a capture at the comparison
            # would write an empty probe file.
            self.sonic_gate_dev = None
            if self.sonic_target is not None and self.leg_target is not None:
                self.sonic_gate_dev = float(np.max(np.abs(
                    np.asarray(self.sonic_target, dtype=np.float64)
                    - np.asarray(self.leg_target, dtype=np.float64))))
            self._sonic_gate_probe()
            if self.sonic_warm < self.sonic_warmup_frames:
                return (f"ERROR:sonic warmup {self.sonic_warm}/"
                        f"{self.sonic_warmup_frames}")
            # A dead stream freezes sonic_warm, so a stale count
            # could enable the lane. Check freshness first.
            # monotonic(), matching sonic_last_rx.
            age = time.monotonic() - self.sonic_last_rx
            if age > self.sonic_stale_ms / 1000.0:
                return (f"ERROR:sonic stream stale ({age:.3f}s > "
                        f"{self.sonic_stale_ms / 1000.0:.3f}s)")
            st = self.lowstate
            if st is not None and SC is not None:
                if SC.tilt_deg(st["quat"]) > self.sonic_enable_tilt_deg:
                    return "ERROR:sonic tilt too large to enable"
            if self.sonic_target is None:
                return "ERROR:sonic no target"
            # the target exists here, so the stashed deviation is valid.
            dev = self.sonic_gate_dev
            if dev is not None:
                if dev > self.sonic_enable_tol_rad:
                    return f"ERROR:sonic target {dev:.3f} rad from legs"
            self.sonic_mode = 'on'
            self.sonic_mm_latch = st["mode_machine"] if st is not None else None
            self.sonic_last_good_t = time.monotonic()
            # arm park: slew to SONIC's arm default on the ramp.
            # DEFAULT_ANGLES_MUJOCO[15:29] is motor order, no remap.
            if (self.sonic_arm_mode == "park" and SC is not None
                    and self.arm is not None):
                with self._dispatch_lock:
                    self.arm_tau[:] = 0.0
                    self.arm.slew_to_vector(
                        np.array(SC.DEFAULT_ANGLES_MUJOCO[15:29],
                                 dtype=np.float32),
                        time.perf_counter(), self.sonic_entry_ramp_s)
            self._sonic_log(f"ON (b={self.sonic_b:.2f}) — SONIC owns legs",
                            force=True)
            return f"SONIC:on {self.sonic_b:.2f}"
        if sub == "off":
            # handback: b ramps to 0, seam restores boot gains.
            # Not a fault; `sonic on` works without a reset.
            self.sonic_mode = 'off'
            self._sonic_log("OFF (operator handback) -> GR00T", force=True)
            return "SONIC:off"
        if sub == "reset":
            if not from_console:
                return "ERROR:sonic reset is console-only"
            # handback first: order matters, because _sonic_revoke latches
            # faulted=True, so the clear below must come after it and a
            # mid-blend reset must not leave mode='on' with b > 0.
            self._sonic_revoke("reset", force=True)
            # drop the frame-step baseline: a reference stuck past
            # sonic_frame_step_max_rad rejects every frame forever. It
            # re-seeds on the next frame; the seam returns early while
            # sonic_target is None.
            self.sonic_target = None
            self.sonic_faulted = False
            self.sonic_fault_cause = "none"
            self.sonic_armed = False
            self.sonic_seq = -1
            # lock: accept path does warm += 1 from the socket
            # thread; a bare 0 can be lost to that read-modify-write.
            with self._dispatch_lock:
                self.sonic_warm = 0
            self._sonic_log("RESET (fault cleared, disarmed, seq cleared, "
                            "frame baseline dropped, handed back to GR00T)",
                            force=True)
            return "SONIC:reset"
        return "ERROR:sonic arm|on|off|reset|status"

    def _sonic_log(self, msg, force=False):
        """Rate-limited (1 Hz) SONIC-lane log — no assert, no exception.
        `force` bypasses the rate limit for state transitions that must always
        be visible (arm/on/off/reset/revoke/handback), the height lane's rule."""
        now = time.monotonic()
        if force or now - self.sonic_log_t >= 1.0:
            self.sonic_log_t = now
            print(f"[sonic] {msg}")

    def _sonic_warm_tick(self, now):
        """Maintain sonic_warm off the ACCEPTED frame stream (spec §4.2).

        sonic_warm is the length of the CURRENT CONSECUTIVE RUN of accepted
        frames.  It counts every accepted frame - armed or not (operator
        directive 2026-09-12, "count warmup before arming too") - and RESTARTS
        at 1 when the gap since the previous accepted frame exceeds
        sonic_stale_ms: the SAME bound the live lane revokes on (see
        _sonic_seam) and that the §2.4 #7 age gate already uses.  So
        `warm >= sonic_warmup_frames` means "no gap in this run would have
        revoked the live lane" - the property warmup exists to establish.  A
        count filled and then frozen by a dead stream can never satisfy it,
        which is why `sonic on` also re-checks freshness at the gate.

        `now` is passed IN rather than read here: the accept path already reads
        the clock for sonic_last_rx, so this compares against the same instant,
        and a test can drive it with a synthetic clock.  An exact-boundary test
        is impossible through the frame path, because feeding a frame re-reads
        time.monotonic() a few microseconds later.

        Degrades, never raises; no assertion is used anywhere in the lane
        (spec §5).  A disabled lane returns before touching anything, and a
        partially-wired harness reads the class-level sonic_lane_enabled =
        False - so this can never turn every frame into "ERROR:sonic frame
        rejected" through the accept path's blanket except.
        """
        if not self.sonic_lane_enabled:
            return
        prev = self.sonic_last_rx
        if prev > 0.0 and (now - prev) > self.sonic_stale_ms / 1000.0:
            self.sonic_warm = 1          # run broke: the silence the live lane revokes on
        elif self.sonic_warm < self.sonic_warmup_frames:
            self.sonic_warm += 1         # saturates

