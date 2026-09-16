#!/usr/bin/env python3
"""vocab_state.py -- the /yolo/classes protocol (v1), pure stdlib, py3.8.

Batch 4 (S3).  Shared by the yolo node (subscriber, switch
OPENBOT_YOLO_CLASSES=1) and the gateway verb ``yolo_classes`` (publisher), so
a bad request is rejected by the SAME rules on both ends
(docs/YOLOE_BATCH4_ARCHITECTURE.md section 3.4).

Request (std_msgs/String JSON object)::

    {"v": 1, "epoch": <int ns>, "seq": <int>, "op": "set"|"reset"|"query",
     "classes": [{"label": "bin", "prompt": "trash can"}, ...],
     "min_conf": 0.35, "origin": "brain"}

Whole-message problems make the request unusable (``error`` set, nothing
applied).  Per-entry problems only reject that entry:
    label is a base class name     -> rejected[label] = 'base'
    label fails LABEL_RE           -> rejected[label] = 'bad_label'
    prompt fails PROMPT_RE         -> rejected[label] = 'bad_prompt'
    duplicate label                -> first wins (later ones dropped)

Dedupe is on (epoch, seq): a NEW epoch (gateway restart) is always fresh;
within one epoch seq <= last applied seq is a duplicate.  The caller decides
when to ``commit`` (the node does not commit a failed swap).

Deployed co-located with yolo_node.py on the box (hand-scp, flat layout).
"""
import json
import math
import re

PROTOCOL_VERSION = 1
MAX_EXTRAS = 236                      # 20 base + up to 236 extras
MIN_CONF_LO = 0.20
MIN_CONF_HI = 0.90
OPS = ('set', 'reset', 'query')
LABEL_RE = re.compile(r'^[a-z][a-z0-9_]{0,31}$')
PROMPT_RE = re.compile(r'^[a-z][a-z0-9 \-]{0,47}$')

# classes_state key order
STATE_KEYS = ('v', 'epoch', 'seq', 'dup', 'backend', 'model', 'base',
              'extras', 'resolved', 'rejected', 'min_conf', 'swap_ms',
              'stamp', 'error')

FIXED_BACKEND_ERROR = 'vocabulary is fixed on this backend'


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))
