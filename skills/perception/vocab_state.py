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


def _key(label):
    """A printable dict key for a rejected entry (never raises)."""
    if isinstance(label, str):
        return label[:48]
    return repr(label)[:48]


def validate(msg, base_names=()):
    """Validate one decoded /yolo/classes request.

    Returns ``(request, error)``.  ``error`` is None or a string
    (``'bad_version'`` / ``'bad_request:<why>'``); on error ``request`` still
    carries whatever ``epoch``/``seq`` could be read (ints or None) so a state
    can echo them.  ``request`` keys: v, epoch, seq, op, classes (list of
    (label, prompt) tuples, base/bad entries removed), rejected
    ({label: reason}), min_conf (float or None), origin.
    """
    req = {'v': None, 'epoch': None, 'seq': None, 'op': None,
           'classes': [], 'rejected': {}, 'min_conf': None, 'origin': None}
    if not isinstance(msg, dict):
        return req, 'bad_request:not_object'
    epoch = msg.get('epoch')
    seq = msg.get('seq')
    if _is_int(epoch):
        req['epoch'] = epoch
    if _is_int(seq):
        req['seq'] = seq
    req['v'] = msg.get('v')
    origin = msg.get('origin')
    req['origin'] = origin if isinstance(origin, str) else None
    if msg.get('v') != PROTOCOL_VERSION or isinstance(msg.get('v'), bool):
        return req, 'bad_version'
    if not _is_int(epoch) or epoch < 0:
        return req, 'bad_request:epoch'
    if not _is_int(seq) or seq < 0:
        return req, 'bad_request:seq'
    op = msg.get('op')
    if op not in OPS:
        return req, 'bad_request:op'
    req['op'] = op
    if 'min_conf' in msg and msg.get('min_conf') is not None:
        mc = msg.get('min_conf')
        if not _is_num(mc) or not (MIN_CONF_LO <= float(mc) <= MIN_CONF_HI):
            return req, 'bad_request:min_conf'
        req['min_conf'] = float(mc)
    if op != 'set':
        return req, None
    classes = msg.get('classes')
    if not isinstance(classes, list):
        return req, 'bad_request:classes'
    if len(classes) > MAX_EXTRAS:
        return req, 'bad_request:too_many_classes'
    base = frozenset(base_names)
    seen = set()
    for entry in classes:
        if not isinstance(entry, dict):
            req['rejected'][_key(entry)] = 'bad_entry'
            continue
        label = entry.get('label')
        prompt = entry.get('prompt')
        if isinstance(label, str) and label in base:
            req['rejected'][label] = 'base'
            continue
        if not isinstance(label, str) or not LABEL_RE.match(label):
            req['rejected'][_key(label)] = 'bad_label'
            continue
        if not isinstance(prompt, str) or not PROMPT_RE.match(prompt):
            req['rejected'][label] = 'bad_prompt'
            continue
        if label in seen:
            continue                       # dup: first wins
        seen.add(label)
        req['classes'].append((label, prompt))
    return req, None


def parse(text, base_names=()):
    """Decode a std_msgs/String payload and validate it (never raises)."""
    try:
        msg = json.loads(text)
    except Exception:  # noqa: BLE001 -- any decode blow-up just means a bad request
        req, _ = validate(None, base_names)
        return req, 'bad_request:json'
    return validate(msg, base_names)


class Dedupe(object):
    """(epoch, seq) duplicate filter.  ``is_dup`` never mutates; the caller
    ``commit``s only what it actually processed."""

    def __init__(self):
        self.last_epoch = None
        self.last_seq = None

    def is_dup(self, epoch, seq):
        if self.last_epoch is None or epoch != self.last_epoch:
            return False
        return seq <= self.last_seq

    def commit(self, epoch, seq):
        if epoch != self.last_epoch:
            self.last_epoch = epoch
        self.last_seq = seq


def build_state(epoch, seq, dup, backend, model, base, extras, resolved,
                rejected, min_conf, swap_ms, stamp, error):
    """One /yolo/classes_state dict with the section-3.4 key order."""
    vals = (PROTOCOL_VERSION, epoch, seq, bool(dup), backend, model,
            list(base), list(extras), dict(resolved), dict(rejected),
            min_conf, swap_ms, stamp, error)
    return dict(zip(STATE_KEYS, vals))


# ── /yolo/strict: gateway leases ──────
# one std_msgs/String JSON per message:
#   narrow:  {"v": 1, "epoch": .., "seq": .., "op": "narrow",
#             "labels": ["bin"], "lease_s": 4.0, "origin": "gateway"}
#   release: {"v": 1, "epoch": .., "seq": .., "op": "release"}
# narrow: only those labels till the lease dies (gateway re-sends
# every second); release or a dead lease restores the full set.
# base names keep their spaces.
STRICT_OPS = ('narrow', 'release')
STRICT_LEASE_LO = 0.5
STRICT_LEASE_HI = 30.0
STRICT_MAX_LABELS = 8


