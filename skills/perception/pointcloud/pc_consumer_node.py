#!/usr/bin/env python3
"""pc_consumer_node: the on-demand point-cloud consumer (batch 3).

DISARMED BY DEFAULT.  While disarmed the node holds exactly one subscription,
/pc_arm (std_msgs/String JSON).  It never subscribes to /object_pointcloud or
/object_detections_body unless armed (or 'warm'), and it destroys both
subscriptions when the task finishes or its TTL (a dead-man) expires.

Request (/pc_arm, JSON):
  {"rid": str, "task": hold_baseline|hold_check|place_rim|split|grasp|person
                       |warm|disarm,
   "label": str, "ref": [x, y]  (MANDATORY for hold_baseline),
   "pose": {"src": "measured", "torso_z", "pitch", "roll", "waist_pitch", "t"},
   "ttl_s": float,
   hold_check only: "baseline_rid": str, "phase": tucked|cancelled|error|other,
                    "t_reply_wall": float (engine reply wall time)}
Result (/pc_result, JSON): exactly ONE envelope per armed request
  (pc_core.envelope), including skip counters and the node's own CPU / RSS.

All logic is in ConsumerCore (no ROS import) so tests drive it directly.  The
rclpy glue is in main() and imports rclpy lazily: without ROS the script exits
with rc 2 and a one-line message, it does not traceback.

Python 3.8-safe.  stdlib + numpy (via pc_core).
"""
import collections
import json
import math
import os
import sys
import time

try:  # normal package import
    from . import pc_core as core
except ImportError:  # flat import: python3 pc_consumer_node.py
    import pc_core as core

TOPIC_ARM = "/pc_arm"
TOPIC_RESULT = "/pc_result"
TOPIC_CLOUD = "/object_pointcloud"
TOPIC_BODY = "/object_detections_body"


def _quota(task, cfg):
    return {
        "hold_baseline": cfg.HOLD_BASE_FRAMES,
        "hold_check": cfg.HOLD_CHK_FRAMES,
        "place_rim": cfg.BOWL_FRAMES,
        "split": 1,
        "grasp": 1,
        "person": cfg.PERSON_POOL_N,
    }[task]


def _rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576.0
    except Exception:  # noqa: BLE001 - not Linux, so it's unknown
        return 0.0


class ConsumerCore(object):
    def __init__(self, cfg=None, wall=time.time, rss_fn=_rss_mb, cpu_fn=time.process_time,
                 perf=time.perf_counter):
        self.cfg = cfg or core.DEFAULT_CFG
        self._wall = wall
        self._rss = rss_fn
        self._cpu = cpu_fn
        self._perf = perf
        self.state = "disarmed"
        self.req = None
        self.deadline = None
        self.bodies = collections.deque(maxlen=2)
        self.frames = collections.deque(maxlen=1)
        self.baselines = collections.OrderedDict()   # rid -> (t_mono, baseline)
        self.out = []
        self._busy = False
        self._last_proc = -math.inf
        self._cooldown = 0
        self._cpu0 = 0.0
        self._reset_counters()

    # ------------------------------------------------------------ small helpers
    def _reset_counters(self):
        self.skipped = {"rate": 0, "busy": 0, "budget": 0, "unpaired": 0, "format": 0,
                        "early": 0, "body_parse": 0}
        self.max_proc_s = 0.0

    def subscriptions(self):
        if self.state == "disarmed":
            return frozenset()
        return frozenset({TOPIC_CLOUD, TOPIC_BODY})

    def pop_results(self):
        r, self.out = self.out, []
        return r

    def _emit_immediate(self, rid, task, res, req):
        env = core.envelope(rid, task, res, req, [], self.cfg, skipped={}, proc_cpu_s=0.0,
                            rss_mb=round(self._rss(), 1), max_proc_s=0.0)
        self.out.append(env)

    def _disarm(self):
        self.state = "disarmed"
        self.req = None
        self.deadline = None
        self.frames = collections.deque(maxlen=1)
        self.bodies.clear()
        self._cooldown = 0

    # ------------------------------------------------------------ stuff coming in
    def on_arm(self, text, t_mono):
        try:
            req = json.loads(text)
            if not isinstance(req, dict):
                raise ValueError("not an object")
        except ValueError:
            self._emit_immediate(None, None, core.result(core.ERROR, core.V_ERROR, "bad_json"), {})
            return
        rid = req.get("rid")
        task = req.get("task")
        if not isinstance(rid, str) or not rid:
            self._emit_immediate(None, task, core.result(core.ERROR, core.V_ERROR, "no_rid"), req)
            return
        if task == "disarm":
            if self.state == "armed" and self.req.get("rid") == rid:
                self._finish("disarmed_by_caller", t_mono=t_mono)
            elif self.state == "warm":
                self._disarm()
            return
        if task == "warm":
            if self.state == "disarmed":
                ttl = self._ttl(req, self.cfg.NODE_WARM_TTL_MAX_S)
                self.state = "warm"
                self.deadline = t_mono + ttl
            return
        if task not in core.TASKS:
            self._emit_immediate(rid, task, core.result(core.ERROR, core.V_ERROR, "bad_task"), req)
            return
        if self.state == "armed":
            if self.req.get("rid") != rid:
                self._emit_immediate(rid, task, core.result(core.ERROR, core.V_ERROR, "busy"), req)
            return
        pose = req.get("pose")
        if task in ("hold_baseline", "hold_check", "place_rim") and isinstance(pose, dict) \
                and "t" in pose:
            try:
                age = self._wall() - float(pose["t"])
            except (TypeError, ValueError):
                age = math.inf
            if age > self.cfg.POSE_MAX_AGE_S or age < -self.cfg.POSE_MAX_AGE_S:
                self._emit_immediate(rid, task, core.unknown("pose_stale"), req)
                return
        if task == "hold_check" and req.get("baseline_rid") not in self.baselines:
            self._emit_immediate(rid, task, core.unknown("no_baseline"), req)
            return
        self.req = req
        self.state = "armed"
        self.deadline = t_mono + self._ttl(req, self.cfg.NODE_TTL_MAX_S)
        self.frames = collections.deque(maxlen=_quota(task, self.cfg))
        self._reset_counters()
        self._cpu0 = self._cpu()

    def _ttl(self, req, cap):
        try:
            ttl = float(req.get("ttl_s", cap))
        except (TypeError, ValueError):
            ttl = cap
        if not math.isfinite(ttl):
            ttl = cap
        return min(max(ttl, self.cfg.NODE_TTL_MIN_S), cap)

    def on_body(self, text, t_mono):
        if self.state == "disarmed":
            return
        try:
            rows = core.parse_body(text)
        except ValueError:
            self.skipped["body_parse"] += 1
            return
        self.bodies.appendleft((t_mono, rows))   # newest first

    def on_cloud(self, data, width, point_step, fields, is_bigendian, hdr_stamp_ns,
                 t_mono, t_wall=None, height=1):
        if self.state != "armed":
            return
        if self._busy:
            self.skipped["busy"] += 1
            return
        if t_mono - self._last_proc < self.cfg.NODE_MIN_PERIOD_S:
            self.skipped["rate"] += 1
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            self.skipped["budget"] += 1
            return
        self._busy = True
        p0 = self._perf()
        t_wall = self._wall() if t_wall is None else t_wall
        try:
            try:
                pts, oid, _nf = core.decode_cloud(data, width, point_step, fields, is_bigendian,
                                                  height)
            except core.CloudFormatError:
                self.skipped["format"] += 1
                return
            fr, _why = core.pair_cloud(list(self.bodies), pts, oid, t_mono, self.cfg,
                                       t_wall=t_wall, n_raw=int(width) * int(height),
                                       hdr_stamp_ns=hdr_stamp_ns)
            if fr is None:
                self.skipped["unpaired"] += 1
                return
            if self.req["task"] == "hold_check":
                try:
                    t_open = float(self.req.get("t_reply_wall")) + self.cfg.HOLD_CHK_DELAY_S
                except (TypeError, ValueError):
                    t_open = math.inf
                if fr.t_wall < t_open:
                    self.skipped["early"] += 1
                    return
            self.frames.append(fr)
        finally:
            dt = self._perf() - p0
            self.max_proc_s = max(self.max_proc_s, dt)
            self._last_proc = t_mono
            self._busy = False
            if dt > self.cfg.NODE_CB_BUDGET_S:
                self._cooldown = int(math.ceil(dt / self.cfg.NODE_CB_BUDGET_S))
        if self.state == "armed" and len(self.frames) >= self.frames.maxlen:
            self._finish(None, t_mono=t_mono)

