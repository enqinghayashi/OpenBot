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

