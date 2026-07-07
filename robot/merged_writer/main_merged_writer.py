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

# ------------------------------------------------------------- the SONIC lane
# optional sonic_core tables: gain tables and joint order.
# static data only, no onnxruntime; a failed import leaves the
# lane absent, never half-present, and init forces it off.
_SONIC_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sonic")
if os.path.isdir(_SONIC_DIR):
    sys.path.insert(0, _SONIC_DIR)
try:
    import sonic_core as SC
except Exception as _e:            # noqa: BLE001 - just degrade, never raise
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

PosStopF = 2146000000.0  # hg position-stop sentinel
VelStopF = 16000.0  # hg velocity-stop sentinel
MODE_ENABLE = 1  # 1 = enable, 0 = disable

# ankle roll per hip roll, keeps the sole tilt fixed.
# foot roll is ~linear in both joints, so
#  d(ankle)/d(hip) = -(d roll/d hip) / (d roll/d ankle).
# the ratio is pose dependent, about -0.80 to -0.91 from
# standing to deep squat, so -0.85 is a mid default.
# sole tilt is ~1 deg per deg of ankle roll: a 6 deg splay
# gives ~0.5 deg of tilt. override with stance_ankle_ratio.
STANCE_ANKLE_RATIO = -0.85

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
    # ------------------------------------------------------------- SONIC lane
    # class-level defaults so a half-wired instance reads the lane as off
    # instead of raising AttributeError at 250 Hz.
    # keep in sync with _init_sonic_state.
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
    # enable-gate probe: an empty string means off.
    # the literals must match _init_sonic_state.
    sonic_gate_probe_dir = ""
    sonic_gate_probe_max = 512
    sonic_gate_probe_fh = None
    sonic_gate_probe_path = None
    sonic_gate_probe_n = 0
    sonic_gate_dev = None

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
        self.payload = False  # payload on/off
        self.rc_hold = False  # pick interlock
        self.cmd_auto = np.zeros(3, dtype=np.float32)  # autopilot, latched
        self.cmd_auto_active = False  # rc ignored
        self.cmd_auto_deadline = 0.0  # adapter watchdog
        self._dispatch_lock = threading.Lock()  # arm mutations
        self._init_height_state()  # height lane
        self._init_sonic_state()  # SONIC lane
        self._init_stance_state()  # stance lane

        self.state = "INIT"
        self._shutdown = False
        self._shutting_down = False
        self._shutdown_cause = "none"
        self.t0 = None  # loop start
        self.loop_count = 0
        self.policy_ticks = 0
        self.prev_tick = None  # drop accounting
        self._overruns = 0
        self._dropped = 0
        self.rc_cmd_zero = np.zeros(3, dtype=np.float32)

        self.lowcmd = None
        self.lowcmd_publisher = None
        self.lowstate = None  # last good state
        self.captured_q = None        # 35-D boot pose from bring-up

        self.policy = None
        self.rc = None
        self.arm = None
        self.obs = None
        self.leg_target = None  # 15-D target
        self.policy_target = None  # GR00T side of the seam

        # shutdown slew runs faster than the 0.3 rad/s pose slew
        # so the arm release finishes inside the tail window.
        self.shutdown_slew_rate = float(self.cfg.get("arm_shutdown_slew_rate", 0.8))
        self.shutdown_slew_min = float(self.cfg.get("arm_shutdown_slew_min_s", 0.5))

        self._simulate_fatal = False  # fatal test hook

    # ------------------------------------------------------------- boot-up stuff

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
        self.lowcmd.mode_pr = 0  # Mode.PR = series
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
        # boot gains for legs and waist, snapshotted after the writes
        # above so waist_pitch_kd_bias is included.
        # b == 0 restores them bit-exactly with no config re-read.
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
            # non-tty stdin means the launcher already gated the run;
            # arm immediately instead of dying on EOF.
            print("[gate] stdin not a tty (launcher-gated) — arming immediately")

    # ------------------------------------------------------------- the console

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

    # ---------------------------------------------- the socket server

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
            prev_reply = None  # previous reply on this connection
            try:
                for raw in conn_file:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        reply = self.handle_socket_line(line, conn)
                    except Exception as e:
                        reply = f"ERROR:{e}"
                    # log suppression: skip the echo for the 50 Hz ack and for
                    # byte-identical repeats; every other reply prints.
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
                self.arm_tau[:] = 0.0  # pose slew, no FF
                self.arm.slew_to(pose, now)
            return "DONE"  # slew started
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
                # optional arm feedforward: 14 tau values applied to motors
                # 15-28 on every tick for the whole slew.
                # no tau given means zero feedforward.
                try:
                    tau = np.array([float(x) for x in parts[16:30]],
                                   dtype=np.float64)
                except ValueError:
                    return "ERROR:arm tau must be 14 floats"
            with self._dispatch_lock:
                self.arm_tau[:] = tau if tau is not None else 0.0
                self.arm.slew_to_vector(target, now, duration)
            return "DONE"  # slew started
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
        if verb == "sonic":
            # these verbs are console-gated by the sonic dispatcher;
            # anything else is an 18-token 50 Hz frame.
            if len(parts) == 1:
                return f"SONIC:status {self._sonic_status()}"
            if parts[1] in ("arm", "on", "off", "reset", "status"):
                return self._apply_sonic(parts, from_console=False)
            # --- 18-token frame, checked in order ------
            # the bare except is required: a bad float raises ValueError
            # while run() only catches RuntimeError.
            # every reject bumps sonic_rej and replies ERROR:.
            try:
                if len(parts) != 18:
                    self.sonic_rej += 1
                    return "ERROR:sonic frame needs 18 tokens"
                vals = [float(x) for x in parts[1:18]]  # 17 numeric fields
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
                # the negation matters: the predicate is age <= stale, so the
                # reject is its logical negation.
                # `age > stale` would pass a NaN age (both comparisons false)
                # and the revoke would never fire.
                if not (age_ms <= self.sonic_stale_ms):
                    self.sonic_rej += 1
                    return "ERROR:state age too old"
                with self._dispatch_lock:
                    # warmup counts consecutive accepted frames, not frames since
                    # `arm`: the satellite streams for minutes, so gating on the
                    # commit would break "watch rx rise, then arm".
                    # the run restarts at 1 when the gap exceeds sonic_stale_ms,
                    # the bound shared by the live revoke and the age gate, so
                    # warm >= frames means no gap in this run would have revoked.
                    # clock: read once and rebind, never slice.
                    now = time.monotonic()
                    self._sonic_warm_tick(now)
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
            # estop: b goes to 0 and the GR00T gains are written immediately.
            # this returns control to GR00T and never requests shutdown,
            # so the body is never left limp mid-walk.
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
                    f"{self._stance_status()} {self._sonic_status()}")
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

    # ---------------------------------------------- runtime height lane
    # the height verb drives an observation input (obs f[3]) rather than
    # a velocity, so it is not orthogonal to cmd.
    # rc_hold forces rc_live false, so the stick read is unreachable:
    # a push pauses a squat but cannot abort it.
    # the watchdog returns to height_stand, never to 0.

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
        if self.height_min > self.height_max:  # tolerate an inverted band
            self.height_min, self.height_max = self.height_max, self.height_min
        self.height_slew_rate = float(self.cfg.get("height_slew_rate", 0.15))
        self.height_hold_s = float(self.cfg.get("height_hold_s", 5.0))
        self.height_ascend_timeout = float(
            self.cfg.get("height_ascend_timeout_s", 3.0))
        self.height_ascend_tol = float(
            self.cfg.get("height_ascend_tol_rad", 0.25))
        self.height_rc_override = float(
            self.cfg.get("height_rc_override_thresh", 0.1))
        # hysteresis: suspend at or above the override threshold and
        # resume at or below this half threshold, so the lane cannot flap.
        self.height_rc_resume = float(
            self.cfg.get("height_rc_resume_thresh",
                         0.5 * self.height_rc_override))
        self.height_stand_eps = 1e-3
        self.height_ref = self.height_stand  # obs f[3]
        self.height_target = self.height_stand  # requested, clamped
        self.height_active = False  # latch + deadline
        self.height_deadline = 0.0
        # a suspend is not a cancel: the target and its deadline are stowed
        # here and restored when the sticks go neutral.
        # _return_to_standing drops the stash.
        self.height_suspended = False
        self.height_suspend_target = self.height_stand
        self.height_suspend_deadline = 0.0
        self._height_log_t = 0.0
        # standing leg q: hip -0.4122/-0.4148, knee 0.7354/0.7404,
        # ankle -0.3184/-0.3211 at z=0.747 with std 0.006.
        # the yaml defaults miss 0.44 rad at the knee against a 0.25 rad
        # tolerance, so the ascend check never passed; take the pose from
        # the policy_stand_q_legs config key instead.
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
        # fresh squat wins
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

    # ------------------------------------------------ additive stance lane
    # symmetric hip-roll splay set at runtime: no deadline, no watchdog,
    # no thread. the failure mode is benign, a wide stance held through a
    # descend, pick and retreat.
    # sticky until `stance off` or a restart, and never persisted.

    def _init_stance_state(self):
        """Stance-lane state.  SAFE BY DEFAULT: the band ships SHUT, so a config
        that predates this lane cannot splay the legs and `stance <deg>` can
        only ever return to 0.  Arming is a deliberate config edit.

        Every key is read with `.get(..., <safe default>)` for the same reason
        _init_height_state does — an older config must not raise KeyError at
        construction."""
        self.stance_deg_max = float(self.cfg.get("stance_deg_max", 0.0))
        if not np.isfinite(self.stance_deg_max) or self.stance_deg_max < 0.0:
            self.stance_deg_max = 0.0  # NaN or negative disables the band
        self.stance_ankle_ratio = float(
            self.cfg.get("stance_ankle_ratio", STANCE_ANKLE_RATIO))
        if not np.isfinite(self.stance_ankle_ratio):
            self.stance_ankle_ratio = STANCE_ANKLE_RATIO
        self.stance_deg = 0.0  # current splay
        self._stance_base_policy = None  # boot snapshot, never compounded
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

        THE MECHANISM, and why it is an exact reparametrisation rather than a
        trim the policy fights.  policy_runner:50 is
        `target_q = action * action_scale + defaults`, and obs_builder:105 is
        `f[13:42] = q - default_padded`; both read the SAME config key, but
        through two INDEPENDENT np.array copies (policy_runner:19,
        obs_builder:43-46).  Shift BOTH by the same delta d:

            obs    = (q + d) - (default + d) = q - default   -> BIT-IDENTICAL
            target = action*scale + default + d     -> fixed point moves by d

        So the policy's own fixed point translates by exactly d with ZERO
        distribution shift, and the splayed stance is a state the policy holds
        rather than a standing error the PD pulls against.  (Confirmed on
        hardware: a hand-set splay survived a descend, a pick and a retreat,
        which a restoring hip-roll loop at kp=150 could not have allowed.)

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
        # sign map: motor 1 left hip, 5 left ankle, 7 right hip,
        # 11 right ankle. each ankle takes its own leg's sign
        # (7 and 11 negated with 1 and 5), so it follows its hip.
        # the negative ratio counter-rotates the ankle and adds
        # about 1 deg of sole tilt per deg of ankle roll.
        d_pol = np.zeros_like(self._stance_base_policy)
        d_pol[1] = +hip
        d_pol[7] = -hip
        d_pol[5] = +ank
        d_pol[11] = -ank
        d_obs = np.zeros_like(self._stance_base_obs)
        d_obs[0:15] = d_pol  # first 15 slots: legs and waist
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

    # ------------------------------------------------ additive SONIC lane
    # the lane is additive: off with b == 0 yields the same object,
    # the same gains and the same flow.
    # no asserts anywhere, since AssertionError slips past the
    # RuntimeError handlers; every config read uses cfg.get so a
    # missing key degrades instead of raising.

    def _init_sonic_state(self):
        """SONIC-lane state + tunables (spec §3, §6).  FORCES the lane disabled
        when sonic_core failed to import: the lane must degrade to "absent",
        never to "half-present"."""
        self.sonic_lane_enabled = bool(self.cfg.get("sonic_lane_enabled", False))
        if SC is None:
            self.sonic_lane_enabled = False  # no tables, no lane
        # --- the state ---
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
        self.sonic_kp_boot = None  # fill_gains sets it
        self.sonic_kd_boot = None
        # mode_machine changes trigger a revoke, so latch it here.
        self.sonic_mm_latch = None
        # --- tunables, all cfg.get ---
        self.sonic_stale_ms = float(self.cfg.get("sonic_stale_ms", 150))
        self.sonic_entry_ramp_s = float(self.cfg.get("sonic_entry_ramp_s", 1.0))
        self.sonic_handback_ramp_s = float(
            self.cfg.get("sonic_handback_ramp_s", 0.5))
        self.sonic_blend_max_step = float(
            self.cfg.get("sonic_blend_max_step_rad", 0.10))
        self.sonic_warmup_frames = int(self.cfg.get("sonic_warmup_frames", 100))
        # the warm bound is sonic_stale_ms, not a separate window key.
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
        # --- diagnostic probe for the enable gate --------
        # opt-in probe: an empty dir means off, and the default opens no
        # file, creates no directory and changes no reply byte.
        # diagnostic only: it reads lane state and appends JSONL, never
        # writing a gain, target, mode or fault latch.
        # runs on the console thread at human rate, never the rt path.
        self.sonic_gate_probe_dir = str(
            self.cfg.get("sonic_gate_probe_dir", ""))
        self.sonic_gate_probe_max = int(
            self.cfg.get("sonic_gate_probe_max", 512))
        self.sonic_gate_probe_fh = None  # lazy open
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
            # no fault clear here: `sonic reset` is the only clearer, so the
            # sequence is reset, arm, on.
            # clearing here would make the refusal in `on` unreachable and
            # auto-resurrect the lane.
            self.sonic_armed = True
            self._sonic_log(
                "ARMED (console) — fault latched, `sonic reset` first"
                if self.sonic_faulted else "ARMED (console)", force=True)
            return "SONIC:armed"
        if sub == "on":
            if not from_console:
                return "ERROR:sonic on is console-only"
            # live enable: all preconditions must hold.
            if not self.sonic_armed:
                return "ERROR:sonic not armed"
            if self.sonic_faulted:
                return "ERROR:sonic faulted — sonic reset first"
            if not self.sonic_has_frame:
                return "ERROR:sonic no frame received"
            # --- capturing the enable-gate operand ------
            # the operand is max|sonic_target - leg_target|, the scalar the
            # check compares; it is computed once so the record and the
            # comparison agree.
            # captured above the warmup check on purpose: the frame-step gate
            # can starve accept and leave warm stalled at 0 or 1, so `on` would
            # never reach the comparison; capturing below it would always record
            # an empty operand. the check itself is not loosened.
            # probe off: one 15-element subtract, no file, no directory and no
            # reply byte change.
            # probe on: one append plus flush per `sonic on`, console thread
            # only, never the rt path.
            self.sonic_gate_dev = None
            if self.sonic_target is not None and self.leg_target is not None:
                self.sonic_gate_dev = float(np.max(np.abs(
                    np.asarray(self.sonic_target, dtype=np.float64)
                    - np.asarray(self.leg_target, dtype=np.float64))))
            self._sonic_gate_probe()
            if self.sonic_warm < self.sonic_warmup_frames:
                return (f"ERROR:sonic warmup {self.sonic_warm}/"
                        f"{self.sonic_warmup_frames}")
            # freshness gate: warm only restarts on a frame, so a dead stream
            # leaves it frozen at the requirement.
            # without this, arm then on could enable on a stale count.
            # same predicate and clock as the live revoke, checked first.
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
            # the guard tests dev, not leg_target: a None sonic_target
            # returned above, and accept only rebinds non-None, so the
            # compared values stay equal either way.
            dev = self.sonic_gate_dev
            if dev is not None:
                if dev > self.sonic_enable_tol_rad:
                    return f"ERROR:sonic target {dev:.3f} rad from legs"
            self.sonic_mode = 'on'
            self.sonic_mm_latch = st["mode_machine"] if st is not None else None
            self.sonic_last_good_t = time.monotonic()
            # park: slew the arms to SONIC's default pose over the handover
            # ramp. both sides use the same joint order, so no remap is needed.
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
            # handback: b ramps to 0 and the seam restores the boot gains when
            # it lands. this is not a fault.
            self.sonic_mode = 'off'
            self._sonic_log("OFF (operator handback) -> GR00T", force=True)
            return "SONIC:off"
        if sub == "reset":
            if not from_console:
                return "ERROR:sonic reset is console-only"
            # reset hands back first: a reset mid-blend would otherwise clear
            # the fault while keeping authority.
            # _sonic_revoke latches the fault, so the clear below must come
            # after it.
            self._sonic_revoke("reset", force=True)
            # drop the frame baseline: every frame is compared to the last
            # accepted one, so a reference more than sonic_frame_step_max_rad
            # away would reject forever. clearing re-seeds the comparison.
            # safe: None is never dereferenced, the seam bails on it and a None
            # guard covers the rest.
            self.sonic_target = None
            self.sonic_faulted = False
            self.sonic_fault_cause = "none"
            self.sonic_armed = False
            self.sonic_seq = -1
            # under the lock: the accept path updates sonic_warm from the socket
            # thread, so a bare `= 0` can lose an update against `+= 1`.
            # reset wants a cold counter.
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
            self.sonic_warm = 1  # frame run broke, so the lane revokes on this gap
        elif self.sonic_warm < self.sonic_warmup_frames:
            self.sonic_warm += 1  # saturates at the warmup requirement

    def _sonic_gate_probe(self):
        """FORENSIC capture of the `sonic on` ENABLE-GATE operand (spec §8.1
        FINDING-8; operator directive 2026-09-12 "measure, then decide").

        Appends ONE JSONL record per `sonic on` invocation that reaches the
        gate head — armed, unfaulted, and at least one frame ever accepted.
        That INCLUDES invocations the warmup / freshness / tilt / tolerance
        checks are about to REFUSE, which is the point: at shipped constants
        the warm counter can stall, so a capture placed at the tolerance
        comparison never fires and the operator gets an empty file.

        OPT-IN and OFF BY DEFAULT.  With sonic_gate_probe_dir == "" (the
        shipped value) this returns at its first statement: no file, no
        handle, no directory, no reply byte changed.  It is called ONLY from
        the `sonic on` branch, which is console-only (§2.2), so this I/O is
        console-thread and human-command-rate — never on the 250 Hz rt path.

        WHICH REFUSAL FIRED is determined by the record's OWN captured fields,
        with no duplicated gate logic to drift: warm < warmup_frames => the
        warmup refusal; last_rx_age_ms > sonic_stale_ms => the freshness
        refusal; tilt_deg > sonic_enable_tilt_deg => the tilt refusal;
        have_target False => "sonic no target"; dev > tol => the tolerance
        refusal.  The authoritative reply string is the console line printed
        by console_command at the same instant; it is not stored here.

        Degrades on every failure: no assert, never raises, and a failure can
        neither change a reply nor touch lane state.  allow_nan=False is used
        DELIBERATELY: the default (allow_nan=True) does NOT raise, it persists
        the bare token NaN (measured), which is not valid RFC-8259 JSON and is
        rejected by jq / pandas / any non-Python reader.  A non-serialisable
        record is DROPPED and logged, never written as an unparseable line.
        """
        if not self.sonic_gate_probe_dir:
            return
        try:
            if self.sonic_gate_probe_n >= self.sonic_gate_probe_max:
                return
            now = time.monotonic()
            st = self.lowstate
            tg = (None if self.sonic_target is None
                  else np.asarray(self.sonic_target, dtype=np.float64))
            lg = (None if self.leg_target is None
                  else np.asarray(self.leg_target, dtype=np.float64))
            diff = None if (tg is None or lg is None) else (tg - lg)
            rec = {
                "schema": 1,
                "stage": "gate-head",
                "pid": os.getpid(),
                "run_t0": getattr(self, "t0", None),
                "t_mono": now,
                "t_wall": time.time(),
                "dev": (None if self.sonic_gate_dev is None
                        else float(self.sonic_gate_dev)),
                "tol": float(self.sonic_enable_tol_rad),
                "have_target": bool(tg is not None),
                "sonic_target": None if tg is None else tg.tolist(),
                "leg_target": None if lg is None else lg.tolist(),
                "diff": None if diff is None else diff.tolist(),
                "argmax": (None if diff is None
                           else int(np.argmax(np.abs(diff)))),
                "warm": int(self.sonic_warm),
                "warmup_frames": int(self.sonic_warmup_frames),
                "armed": bool(self.sonic_armed),
                "mode": str(self.sonic_mode),
                "b": float(self.sonic_b),
                "faulted": bool(self.sonic_faulted),
                "fault_cause": str(self.sonic_fault_cause),
                "has_frame": bool(self.sonic_has_frame),
                "seq": int(self.sonic_seq),
                "rx": int(self.sonic_rx),
                "rej": int(self.sonic_rej),
                "last_rx_age_ms": (0.0 if self.sonic_last_rx <= 0.0 else
                                   float((now - self.sonic_last_rx) * 1000.0)),
                "tilt_deg": None,
                "mode_machine": None,
                "tick": None,
                "gyro_max_abs": None,
                "dq_leg_max_abs": None,
            }
            # settled-stand: read the same lowstate dict the tilt check uses;
            # still, not rotating, legs not moving.
            # stays null without a lowstate: a null is honest, 0.0 would fake a
            # settled body.
            try:
                if st is not None:
                    rec["mode_machine"] = int(st["mode_machine"])
                    rec["tick"] = int(st["tick"])
                    if SC is not None:
                        rec["tilt_deg"] = float(SC.tilt_deg(st["quat"]))
                    rec["gyro_max_abs"] = float(np.max(np.abs(
                        np.asarray(st["gyro"], dtype=np.float64))))
                    rec["dq_leg_max_abs"] = float(np.max(np.abs(
                        np.asarray(st["dq"][0:15], dtype=np.float64))))
            except Exception:
                pass
            if self.sonic_gate_probe_fh is None:
                os.makedirs(self.sonic_gate_probe_dir, exist_ok=True)
                self.sonic_gate_probe_path = os.path.join(
                    self.sonic_gate_probe_dir,
                    "sonic_gate_probe_%s_pid%d.jsonl" % (
                        time.strftime("%Y%m%d_%H%M%S"), os.getpid()))
                self.sonic_gate_probe_fh = open(
                    self.sonic_gate_probe_path, "a", encoding="utf-8")
            self.sonic_gate_probe_fh.write(
                json.dumps(rec, allow_nan=False) + "\n")
            self.sonic_gate_probe_fh.flush()
            self.sonic_gate_probe_n += 1
        except Exception:
            # a capture failure never changes the reply or lane state.
            # drop the handle so the next attempt reopens the file.
            try:
                self.sonic_gate_probe_fh = None
                self._sonic_log("gate probe: record dropped")
            except Exception:
                pass

    def _sonic_status(self):
        """Inline status fragment, same shape as _cmd_auto_status/_height_status."""
        if not self.sonic_lane_enabled:
            return "sonic=disabled"
        frag = (f"sonic={self.sonic_mode} b={self.sonic_b:.2f} "
                f"armed={int(self.sonic_armed)} rx={self.sonic_rx} "
                f"rej={self.sonic_rej} warm={self.sonic_warm}")
        if self.sonic_faulted:
            frag += f" FAULT({self.sonic_fault_cause})"
        return frag

    def _sonic_blend_step(self, t):
        """Continuous blend authority b (spec §4): 0->1 entry while the lane is
        live, 1->0 handback otherwise.  NEVER a switch, and it SNAPS exactly to
        0.0 / 1.0 so the off path (b == 0.0) stays bit-identical.

        Rate-integrated at the policy tick period (decimation * dt = 20 ms), so
        no extra clock state is needed.  max(ramp, dt) keeps a zero/absurd
        configured ramp from dividing by zero — no assert.

        `self.sonic_b` IS the authority state (spec §3) and is persisted here —
        the ramp integrates from it every tick, so a caller that ignored the
        return value would freeze b at its initial 0.0 and the lane could never
        engage.  Both snaps write the exact 0.0 / 1.0 back, so the off path's
        `b == 0.0` test stays exact.
        """
        dt = self.decimation * self.dt
        live = (self.sonic_mode == 'on' and self.sonic_armed
                and self.sonic_has_frame)
        if live:
            b = self.sonic_b + dt / max(self.sonic_entry_ramp_s, dt)
        else:
            b = self.sonic_b - dt / max(self.sonic_handback_ramp_s, dt)
        if b >= 1.0:
            self.sonic_b = 1.0
        elif b <= 0.0:
            self.sonic_b = 0.0
        else:
            self.sonic_b = float(b)
        return self.sonic_b

    def _sonic_step_clamp(self, raw):
        """Per-policy-tick slew limit on the blended target (spec §4).  Returns
        a NEW array (never mutates `raw`) clamped to sonic_blend_max_step_rad of
        the previous leg_target, so the handover can never step discontinuously.
        """
        out = np.array(raw, dtype=np.float64, copy=True)
        prev = self.leg_target
        if prev is None:
            return out
        prev = np.asarray(prev, dtype=np.float64)
        return np.minimum(np.maximum(out, prev - self.sonic_blend_max_step),
                          prev + self.sonic_blend_max_step)

    def _sonic_write_leg_gains(self, b):
        """THE GAIN SWAP (spec §4.1 — the single highest-value correctness
        point).  kp AND kd ramp with the SAME b from the fill_gains() boot
        snapshot (which already carries waist_pitch_kd_bias) to
        KP_MUJOCO[0:15]/KD_MUJOCO[0:15].  Shipping SONIC's setpoint under GR00T
        gains is a different controller than the one that went green: knee kp
        200 vs 99.098, waist-pitch kp 250 vs 28.501, and a NEGATIVE waist-pitch
        kd.  At b == 0.0 this restores the snapshot bit-exactly (x*1.0 + y*0.0).
        """
        if SC is None or self.sonic_kp_boot is None or self.sonic_kd_boot is None:
            return
        kp = self.sonic_kp_boot * (1.0 - b) + SC.KP_MUJOCO[0:15] * b
        kd = self.sonic_kd_boot * (1.0 - b) + SC.KD_MUJOCO[0:15] * b
        for i in range(15):
            self.lowcmd.motor_cmd[i].kp = float(kp[i])
            self.lowcmd.motor_cmd[i].kd = float(kd[i])

    def _sonic_revoke(self, cause, force=False):
        """The ONE funnel for every abort trigger (spec §4.2/§4.3).  Returns
        authority to the GR00T lane — never a HOLD, never a LIMP, and NEVER
        request_shutdown: a 250 ms socket hiccup must not end in a zero-torque
        limp mid-walk.

        force=False: mode := off, the lane is marked faulted, and b ramps to 0
                     at the handback rate (the seam restores the boot gains when
                     it lands).  No auto-resurrection: a faulted lane needs
                     `sonic reset` then `sonic arm` then `sonic on`, all
                     console-only.
        force=True:  b := 0 instantly and the boot gains are written NOW (estop
                     and request_shutdown).  If b was already 0 the gains are
                     already boot, so nothing is written and the off path stays
                     untouched.
        """
        if not self.sonic_faulted:
            self.sonic_faulted = True
            self.sonic_fault_cause = str(cause)
            self._sonic_log(f"REVOKE ({cause}) -> returning to GR00T",
                            force=True)
        self.sonic_mode = 'off'
        self.sonic_armed = False
        if force:
            was_live = (self.sonic_b != 0.0)
            self.sonic_b = 0.0
            self.sonic_prev_blend = 0.0
            if was_live:
                self._sonic_write_leg_gains(0.0)

    def _sonic_seam(self, t, st, cmd, rc_live, rc_norm):
        """The ONLY control-flow change in the writer (spec §4).  Called at the
        policy tick where `policy_target` was just computed.

        The first two lines are the falsifiable definition of "additive": with
        the lane off AND b == 0 this assigns the SAME OBJECT the policy
        returned — no copy, no arithmetic, no gain write — so the shipped GR00T
        path is bit-identical to the writer without this lane.
        """
        if self.sonic_mode == 'off' and self.sonic_b == 0.0:
            self.leg_target = self.policy_target
            return

        # --- live-tick revoke checks, in evaluation order ---------
        if self.sonic_mode == 'on':
            cause = None
            now = time.monotonic()
            if now - self.sonic_last_rx > self.sonic_stale_ms / 1000.0:
                cause = "stale frame"
            if cause is None and st is not None and SC is not None:
                if SC.tilt_deg(st["quat"]) > self.sonic_tilt_fail_deg:
                    cause = "tilt"
            if cause is None and rc_live and \
                    rc_norm > self.sonic_rc_override_thresh:
                cause = "rc override"
            if cause is None and st is not None and \
                    st["mode_machine"] != self.sonic_mm_latch:
                cause = "mode_machine changed"
            if cause is None and getattr(self.reader, "_frozen", False):
                cause = "reader frozen"
            # overrun window: head counts ticks, base holds _overruns at the
            # window start. one window is 1 s of policy ticks.
            self.sonic_overrun_head += 1
            window = max(int(self.hz / self.decimation), 1)
            if self.sonic_overrun_head >= window:
                if (self._overruns - self.sonic_overrun_base) > \
                        self.sonic_max_overruns_per_s and cause is None:
                    cause = "overrun rate"
                self.sonic_overrun_base = self._overruns
                self.sonic_overrun_head = 0
            if cause is not None:
                self._sonic_revoke(cause)

        # --- a continuous blend, never a switch -----------------------------
        b = self._sonic_blend_step(t)
        if b <= 0.0:
            if self.sonic_prev_blend != 0.0:
                # handback done: boot gains restored and object identity returned
                # to the policy target.
                self._sonic_write_leg_gains(0.0)
                self._sonic_log("handback complete — GR00T gains restored",
                                force=True)
            self.sonic_prev_blend = 0.0
            self.leg_target = self.policy_target
            return
        if self.sonic_target is None:
            # unreachable in practice: fall back to the GR00T target instead
            # of raising.
            self.leg_target = self.policy_target
            return
        self.sonic_prev_blend = b
        raw = self.policy_target * (1.0 - b) + self.sonic_target * b
        self.leg_target = self._sonic_step_clamp(raw)
        self._sonic_write_leg_gains(b)
        self.sonic_last_good_t = time.monotonic()

    def console_command(self, line):
        if not line:
            return
        parts = line.split()
        verb = parts[0]
        now = time.perf_counter()
        if verb in ("tuck", "carry", "ready", "lift", "neutral", "home"):
            pose = "CARRY_TUCK" if verb in ("tuck", "carry") else verb.upper()
            with self._dispatch_lock:
                self.arm.slew_to(pose, now)
            print(f"[arm] slew -> {pose} ({self.arm._T:.1f} s)")
        elif verb == "arm":
            if len(parts) < 15:
                print("[arm] arm needs 14 floats")
                return
            try:
                target = np.array([float(x) for x in parts[1:15]], dtype=np.float32)
            except ValueError:
                print("[arm] arm needs 14 floats")
                return
            duration = None
            if len(parts) >= 16:
                try:
                    duration = float(parts[15])
                except ValueError:
                    print("[arm] duration must be a float")
                    return
            with self._dispatch_lock:
                self.arm.slew_to_vector(target, now, duration)
            print(f"[arm] slew -> RAW ({self.arm._T:.1f} s)")
        elif verb == "freeze":
            with self._dispatch_lock:
                self.arm.freeze(time.perf_counter())
            print("[arm] slew frozen (holding current interpolated target)")
        elif verb == "q":
            st = self.lowstate
            if st is None:
                print("[arm] no lowstate yet")
                return
            print("[q] " + " ".join(f"{v:.4f}" for v in st["q"][15:29]))
        elif verb == "armstate":
            a = self.arm
            print(f"[arm] state={a.state} active={1 if a._T > 0.0 else 0} "
                  f"rem_T={a._T:.3f}")
        elif verb == "hold_rc":
            if len(parts) != 2 or parts[1] not in ("on", "off"):
                print("[rc] hold_rc on|off")
                return
            self.rc_hold = (parts[1] == "on")
            print(f"[rc] hold_rc={self.rc_hold} "
                  f"({'sticks clamped (pick interlock)' if self.rc_hold else 'sticks live'})")
        elif verb == "payload":
            if len(parts) != 2 or parts[1] not in ("on", "off"):
                print("[arm] payload on|off")
                return
            self.payload = (parts[1] == "on")
            print(f"[arm] payload={self.payload} "
                  f"({'shutdown will NOT release the arms' if self.payload else 'shutdown will release to NEUTRAL'})")
        elif verb == "rc0":
            self.rc.zero()
            print("[rc] cmd zeroed")
        elif verb == "cmd_set":
            err = self._apply_cmd_set(parts)
            if err:
                print(f"[cmd] {err}")
                return
            print(f"[cmd] {self._cmd_auto_status()}")
        elif verb == "height":
            err = self._apply_height(parts)
            if err:
                print(f"[height] {err}")
                return
            print(f"[height] {self._height_status()}")
        elif verb == "stance":
            err = self._apply_stance(parts)
            if err:
                print(f"[stance] {err}")
                return
            print(f"[stance] {self._stance_status()}")
        elif verb == "sonic":
            print(self._apply_sonic(parts, from_console=True))
        elif verb == "estop":
            # force release: b goes to 0 and the gains follow.
            # this never requests shutdown.
            self._sonic_revoke("estop", force=True)
            print("[estop] SONIC lane force-released -> GR00T gains restored")
        elif verb == "simulate_fatal":
            self._simulate_fatal = True
            print("[fatal] SIMULATED FATAL armed — next loop tick raises")
        elif verb in ("status", "?"):
            self.print_status()
        elif verb == "quit":
            print("[console] quit requested")
            self.request_shutdown("console quit")
        else:
            print("commands: tuck|carry ready lift neutral home | arm <14f> [dur] | "
                  "freeze | q | armstate | hold_rc on|off | payload on/off | "
                  "cmd_set off|vx vy omega [dur] | height <m> [dur] | "
                  "height off | stance <deg> | stance off | "
                  "sonic arm|on|off|reset|status | estop | "
                  "simulate_fatal | rc0 | status | quit")

    def print_status(self):
        st = self.lowstate
        if st is None:
            print("no lowstate yet")
            return
        key = self.policy.last_key if self.policy else "no-policy"
        cmd = self.rc.cmd if self.rc else np.zeros(3)
        print(f"[status] state={self.state} loop={self.loop_count} "
              f"policy={key} ticks={self.policy_ticks} "
              f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
              f"arm={self.arm.state} mode_machine={st['mode_machine']} "
              f"payload={self.payload} hold={int(self.rc_hold)} "
              f"{self._cmd_auto_status()} {self._height_status()} "
              f"{self._stance_status()} {self._sonic_status()}")

    # -------------------------------------------------------- the control bit

    def request_shutdown(self, cause):
        # SONIC first: b goes to 0 and the boot gains are written right
        # away, so the shutdown tail never sends a SONIC-gain frame.
        # the call order guarantees this, not a change in the hold tick.
        self._sonic_revoke('shutdown', force=True)
        self._shutdown = True
        self._shutdown_cause = cause
        self._shutting_down = True  # rc dead from here
        self.cmd_auto_active = False  # zeroed regardless
        self.cmd_auto[:] = 0.0
        # height: every abort path drops the latch, so phase 1 drives the
        # legs back up first.
        self.height_active = False
        self.height_target = self.height_stand
        self.height_suspended = False  # no stowed squat
        # stance: zero the offset before the ascend so the ascend check
        # measures the real pose, not a splay that is just inside tol.
        if self._stance_base_policy is not None:
            self._write_stance(0.0)

    def _status_line(self):
        if self.loop_count % (self.hz * 1) != 0:
            return
        st = self.lowstate
        if st is None:
            return
        key = self.policy.last_key if self.policy else "hold"
        cmd = self.rc.cmd if self.rc else np.zeros(3)
        print(f"[t={time.perf_counter()-self.t0:6.1f}] {key} "
              f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
              f"arm={self.arm.state} mode={st['mode_machine']} "
              f"Tmax={st['max_temp']:.0f}C "
              f"poll_to={self.reader.timeout_count} "
              f"overruns={self._overruns} drops={self._dropped} "
              f"hold={int(self.rc_hold)} {self._cmd_auto_status()} "
              f"{self._height_status()} {self._sonic_status()}")

    def run(self):
        print("=== C3 merged whole-body writer ===")
        print(f"stage={self.args.stage} hz={self.hz} policy_dec={self.decimation} "
              f"policy_enabled={self.policy_enabled} rc_enabled={self.rc_enabled}")
        print("=== EXIT BEHAVIOR: ZeroCmd limp (mode 0x01, kp/kd=0, q=PosStopF) "
              "then SelectMode(\"ai\") handoff — sport control restored, RC "
              "ladder live (L1+B -> L2+UP -> R2+A). ===")

        self.acquire_lock()
        print(f"[lock] {self.cfg['lockfile']} held by pid {os.getpid()}")

        if self.args.interface:
            ChannelFactoryInitialize(0, self.args.interface)
        else:
            ChannelFactoryInitialize(0, self.cfg["interface"])
        self.release_mode()

        # --- state reader, first lowstate, capture the pose ---
        self.reader = StateReader(self.cfg)
        self.lowstate = self.reader.wait_first(self.cfg["lowstate_wait_s"])
        self.captured_q = self.lowstate["q"].copy()
        print(f"[bring-up] first lowstate: tick={self.lowstate['tick']} "
              f"mode_machine={self.lowstate['mode_machine']}")
        print(f"[bring-up] captured q: legs {self.captured_q[0:6].round(3)} ... "
              f"arms {self.captured_q[15:22].round(3)}")

        self.init_lowcmd()
        self.fill_gains()

        # --- the subsystems ---
        self.arm = ArmGuard(self.cfg)
        self.arm.capture(self.captured_q, 0.0)
        self.arm_tau = np.zeros(14)  # arm FF, socket-set
        self.obs = ObsBuilder(self.cfg)
        if self.policy_enabled:
            self.policy = PolicyRunner(self.cfg, self.onnx_dir)
            self.leg_target = self.policy.defaults.copy()
            # GR00T-side blend input for the seam, needed before the first
            # seam call.
            self.policy_target = self.leg_target.copy()
            print(f"[policy] loaded {os.path.basename(self.onnx_dir)}")
        else:
            self.leg_target = self.captured_q[:15].copy()
            self.policy_target = self.leg_target.copy()
        # snapshot stance base after both encodings
        # exist, before any verb can touch them.
        self._capture_stance_base()
        if self.rc_enabled:
            self.rc = RCTarget(self.cfg)

        # gate joint hold
        self.gate_q = float(self.cfg["gate_q"])

        self.gate_prompt()

        # --- boot arm pose: slew to NEUTRAL rather than holding the
        # captured pose. minimum 1.5 s, the same slew as any command.
        self.arm.slew_to("NEUTRAL", time.perf_counter())

        # --- hand off to the loop; signal handlers ---
        def _sig(signum, frame):
            print(f"\n[signal] {signum} — shutdown")
            self.request_shutdown(f"signal {signum}")
        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
        self.start_console()
        self.start_socket_server()

        # --- the write loop ---
        self.state = "RUNNING"
        self.t0 = time.perf_counter()
        next_deadline = self.t0
        try:
            while not self._shutdown:
                t = time.perf_counter()
                self._tick(t)
                next_deadline += self.dt
                sleep = next_deadline - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    self._overruns += 1
        except RuntimeError as e:
            print(f"[fatal] {e}")
            self.request_shutdown(f"runtime error: {e}")
        finally:
            self.shutdown()

    def _tick(self, t):
        # the tick counter advances here rather than in the main loop: the
        # first shutdown phase calls _tick from its own settle loop, where a
        # counter fed by the main loop would freeze.
        # the policy gate is `% decimation == 0`, so a frozen count would
        # run the policy never or at 250 Hz instead of 50 Hz.
        # advancing here fires it within `decimation` ticks on every path.
        self.loop_count += 1
        if self._simulate_fatal:
            # one-shot: fires once and clears itself, because _tick runs again
            # in the settle loop where a latched flag would raise a second time.
            self._simulate_fatal = False
            raise RuntimeError("SIMULATED FATAL (test hook)")
        st = self.reader.poll()
        if st is not None:
            self.lowstate = st
            # drop accounting: tick rises at motor rate
            if self.prev_tick is not None:
                gap = st["tick"] - self.prev_tick
                if gap > 1:
                    self._dropped += gap - 1
            self.prev_tick = st["tick"]

        if self.lowstate is None:
            return  # haven't armed yet

        st = self.lowstate
        q, dq = st["q"], st["dq"]

        # command precedence: rc_hold clamps the sticks for the pick
        # window. rc_hold or shutdown gives zero (stationarity, no
        # autopilot, no RC), else cmd_auto, else live RC with raw
        # vx/vy/omega.
        # cmd_auto's monotonic deadline still falls through to RC or zero
        # if the adapter dies.
        if self.cmd_auto_active and time.monotonic() >= self.cmd_auto_deadline:
            self.cmd_auto_active = False
            self.cmd_auto[:] = 0.0
        # height watchdog: same shape as cmd_auto but it recovers to
        # height_stand, never to zero, which is not a safe height.
        if self.height_active and time.monotonic() >= self.height_deadline:
            self.height_active = False
            self.height_target = self.height_stand
            self._height_log("watchdog expiry -> standing")
        if self._shutting_down or self.rc_hold:
            cmd = self.rc_cmd_zero  # all sources dead
            rc_live = False
        elif self.cmd_auto_active:
            cmd = self.cmd_auto  # rc ignored
            rc_live = False
        elif self.rc:
            cmd = self.rc.update(st["wireless_remote"])
            rc_live = True
        else:
            cmd = self.rc_cmd_zero
            rc_live = False

        # height override: a stick can never abort a squat, only pause it.
        # suspend is gated on rc_live and the rc_hold branch forces that
        # false, so the stick read is unreachable while the body is held.
        # the pick engine sets rc_hold before the arm motion and releases
        # it in its own tail, so a hard kill leaves the clamp on past
        # height_hold_s; `hold_rc off` clears it.
        # `not rc_live` resumes a suspended squat as soon as rc_hold opens.
        # a pause restores both the target and its original deadline.
        # resume uses half the suspend threshold as hysteresis, and an
        # expired hold stays cancelled.
        if not (self._shutting_down or self._shutdown):
            rc_norm = float(np.linalg.norm(cmd)) if rc_live else 0.0
            if (self.height_active and rc_live
                    and rc_norm >= self.height_rc_override):
                self.height_suspended = True
                self.height_suspend_target = self.height_target
                self.height_suspend_deadline = self.height_deadline
                self.height_active = False
                self.height_target = self.height_stand
                self._height_log(
                    f"rc override (|rc|={rc_norm:.2f} >= "
                    f"{self.height_rc_override:.2f}) — squat SUSPENDED, "
                    f"reference -> standing {self.height_stand:.2f}; resumes "
                    f"when the sticks are neutral before the hold deadline",
                    force=True)
            elif (self.height_suspended
                  and (not rc_live or rc_norm <= self.height_rc_resume)):
                if time.monotonic() < self.height_suspend_deadline:
                    self.height_target = self.height_suspend_target
                    self.height_deadline = self.height_suspend_deadline
                    self.height_active = True
                    self._height_log(
                        f"sticks neutral — squat RESUMED -> "
                        f"{self.height_target:.2f} "
                        f"({max(self.height_deadline - time.monotonic(), 0.0):.1f}s "
                        f"left in the hold)", force=True)
                else:
                    self._height_log(
                        "sticks neutral — the hold deadline expired while "
                        "suspended: squat CANCELLED (re-issue 'height <m>')",
                        force=True)
                self.height_suspended = False

        # --- runtime height lane here: rate limit plus the obs setter ---
        # rate limited: the reference cannot step, and this runs before the
        # policy tick so obs f[3] reads the same frame.
        hstep = self.height_slew_rate * self.dt
        if self.height_ref < self.height_target:
            self.height_ref = min(self.height_ref + hstep, self.height_target)
        elif self.height_ref > self.height_target:
            self.height_ref = max(self.height_ref - hstep, self.height_target)
        if self.obs is not None:
            self.obs.set_height(self.height_ref)

        # --- the inline 50 Hz policy tick ---
        if self.policy is not None and self.loop_count % self.decimation == 0:
            obs = self.obs.push(q, dq, st["quat"], st["gyro"], cmd, self.policy.action)
            self.policy_target = self.policy.infer(obs, cmd)
            # seam: the one control-flow change. rc_live and rc_norm are
            # resolved above; with the lane off and b == 0 it returns the same
            # object the policy produced.
            rc_norm = float(np.linalg.norm(cmd)) if rc_live else 0.0
            self._sonic_seam(t, st, cmd, rc_live, rc_norm)
            self.policy_ticks += 1
            if self.verbose and self.policy_ticks % 50 == 0:
                print(f"[policy] {self.policy.last_key} "
                      f"target {self.leg_target[0:3].round(3)}")

        # --- coupling ramp: captured q -> targets across 0.12 s ---
        ramp_t = min((t - self.t0) / self.ramp_s, 1.0) if self.t0 else 1.0

        # --- arms (250 Hz, cosine slew, decoupled from the policy tick) ---
        arm_target = self.arm.tick(t)

        # --- put motor_cmd together ---
        lc = self.lowcmd
        lc.mode_machine = st["mode_machine"]          # echo it back
        lc.mode_pr = 0
        for i in range(15):
            lc.motor_cmd[i].q = self.captured_q[i] + \
                (self.leg_target[i] - self.captured_q[i]) * ramp_t
            lc.motor_cmd[i].dq = 0.0
            lc.motor_cmd[i].tau = 0.0
        for j, i in enumerate(range(15, 29)):
            lc.motor_cmd[i].q = arm_target[j]
            lc.motor_cmd[i].dq = 0.0
            lc.motor_cmd[i].tau = self.arm_tau[j]  # gravity FF
        lc.motor_cmd[29].q = self.gate_q
        lc.motor_cmd[29].dq = 0.0
        lc.motor_cmd[29].tau = 0.0
        for i in range(30, 35):
            lc.motor_cmd[i].q = self.captured_q[i]  # captured hold
            lc.motor_cmd[i].dq = 0.0
            lc.motor_cmd[i].tau = 0.0

        lc.crc = self.crc.Crc(lc)
        self.lowcmd_publisher.Write(lc)

        self._status_line()

    # --------------------------------------------------------- shutdown time

