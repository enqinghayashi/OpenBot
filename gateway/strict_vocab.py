"""strict_vocab.py -- B8i strict detection: the gateway's open HOLDS.

stdlib only, py3.8, rclpy-free (unit-testable).  A HOLD narrows the yolo
node's active vocabulary to a few labels for ONE bounded phase:

  navigate               opened by GatewayNode._do_navigate; closed when the
                         verb has returned AND the walk is no longer
                         SEARCHING / DETECTED / APPROACHING, on cancel/stop,
                         or at NAV_HOLD_MAX_S;
  look_for, place, pick  opened by the brain (verb yolo_strict op 'begin');
                         closed by op 'end', on cancel/stop, or at
                         BRAIN_HOLD_MAX_S (a brain that died never ends it).

The node runs the UNION of every open hold's labels plus KEEP_LABELS; with
no hold open it runs the operator's full vocabulary, which only the node
holds and strict detection never edits.  This module decides WHAT the
gateway publishes on /yolo/strict; the node owns the swap and the lease
(LEASE_S, re-sent every RENEW_S: a dead gateway restores the node by itself).
"""
import itertools
import math
import os
import threading
import time

KEEP_LABELS = ('person',)     # operator 2026-10-09: "keep person"
LEASE_S = 4.0                 # ~3 misses = drop
RENEW_S = 1.0                 # gateway renew rate
ACK_S = 1.0                   # node echo wait
NAV_HOLD_MAX_S = 135.0        # NAV_WAIT_S 115 + coordinator 120 s + margin
BRAIN_HOLD_MAX_S = 15.0       # one target resolve
MAX_LABELS = 8                # = vocab_state.STRICT_MAX_LABELS
BRAIN_OWNERS = ('look_for', 'place', 'pick')
#: keys off the gateway's OWN stamps, not grip refresh.  begin: a packet
#: landed SETTLE_S after the swap (re-mint needs CONFIRM_SECONDS 1.0 s +
#: one 0.1 s tick + projector p99 0.027 s + p95 gap 0.25 s).  end: a
#: packet landed REFILL_S after the restore.
SETTLE_S = 1.5
SETTLE_MAX_S = 3.0
REFILL_S = 1.5
REFILL_MAX_S = 2.5
POLL_S = 0.05


class StrictHolds(object):
    """The open holds.  Thread-safe; ``clock`` is injectable (monotonic)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._holds = {}          # hid -> hold dict

    def open(self, owner, labels, max_s):
        """Open a hold for ``labels`` (order kept, duplicates dropped);
        returns its id (int >= 1)."""
        labs = []
        for lab in labels:
            if isinstance(lab, str) and lab and lab not in labs:
                labs.append(lab)
        with self._lock:
            hid = next(self._ids)
            self._holds[hid] = {'owner': str(owner), 'labels': tuple(labs),
                                'deadline': self._clock() + float(max_s),
                                'done': False}
        return hid

    def close(self, hid):
        """Close one hold; False when it was not open."""
        with self._lock:
            return self._holds.pop(hid, None) is not None

    def close_all(self):
        """Close every hold; returns how many were open."""
        with self._lock:
            n = len(self._holds)
            self._holds.clear()
        return n

    def mark_done(self, hid):
        """navigate: the verb returned; the hold now ends with the walk."""
        with self._lock:
            h = self._holds.get(hid)
            if h is None:
                return False
            h['done'] = True
            return True

    def expire(self, walk_active):
        """Close the holds past their cap, and the done navigate holds once
        the walk is inactive.  Returns [(hid, owner, why)]."""
        now = self._clock()
        out = []
        with self._lock:
            for hid in sorted(self._holds):
                h = self._holds[hid]
                if now >= h['deadline']:
                    out.append((hid, h['owner'], 'cap'))
                elif h['done'] and not walk_active:
                    out.append((hid, h['owner'], 'walk_inactive'))
            for hid, _o, _w in out:
                del self._holds[hid]
        return out

    def labels(self):
        """The union of the open holds' labels (oldest hold first) plus
        KEEP_LABELS, at most MAX_LABELS; () when no hold is open."""
        with self._lock:
            out = []
            for hid in sorted(self._holds):
                for lab in self._holds[hid]['labels']:
                    if lab not in out:
                        out.append(lab)
        if not out:
            return ()
        keep = [k for k in KEEP_LABELS if k not in out]
        out = out[:MAX_LABELS - len(keep)] + keep
        return tuple(out)

