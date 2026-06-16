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

