#!/usr/bin/env python3
"""yoloe_backend.py -- YOLOE open-vocabulary detector, BANK-ONLY (batch 4, S2).

Used by yolo_node_SEG_version.py ONLY when ``OPENBOT_YOLOE=1``.  Never runs
the text encoder: the vocabulary is a precomputed embedding bank (NPZ) built
on the laptop (docs/YOLOE_BATCH4_ARCHITECTURE.md D2, section 7), so the box
needs no ``clip``, no 600 MB text-encoder asset and no internet.  A prompt
that is not in the bank is rejected ``not_in_bank`` (there is no encoder
tier).

Bank NPZ keys: ``prompts`` (str array), ``embeds`` (float32 [N,512] or
[1,N,512]), ``model`` (str), ``ultralytics`` (str), ``built`` (str).  The
base vocabulary entries are ``(label=TRACKED name verbatim, prompt=TRACKED
name)`` -- spaces kept ('cell phone'); a bank missing any base prompt is
refused.

``load()`` checks ``os.path.isfile`` on the weights and the bank BEFORE any
constructor runs: a missing ``.pt`` whose basename matches a release asset
otherwise sends ultralytics into a download attempt (network timeouts on a
box without internet).

Every model vocabulary change goes through ``set_classes(names, embeddings)``
with TWO arguments (the one-argument form invokes the text encoder; it is
grep-pinned absent from skills/perception/*.py).

Stdlib + numpy at import time; torch / ultralytics are imported lazily inside
``load()`` (both injectable for tests).  py3.8.
"""
import os
import time

import numpy as np


class BackendUnavailable(Exception):
    """The YOLOE backend cannot run; the node falls back to the engine."""


class VocabResult(object):
    """Outcome of one ``set_vocab`` call."""

    __slots__ = ('names', 'resolved', 'rejected', 'swap_ms', 'changed')

    def __init__(self, names, resolved, rejected, swap_ms, changed):
        self.names = names            # prompt idx -> label
        self.resolved = resolved      # {label: 'bank'}
        self.rejected = rejected      # {label: reason}
        self.swap_ms = swap_ms        # ms, 0.0 if unchanged
        self.changed = changed        # did we touch the model?


def _model_stem(name):
    """'.../yoloe-26n-seg.pt' / '.../yoloe-26n-seg.engine' / 'yoloe-26n-seg' / 'yoloe-26n' -> 'yoloe-26n'."""
    s = os.path.basename(str(name)).strip().lower()
    if s.endswith('.pt'):
        s = s[:-3]
    elif s.endswith('.engine'):
        s = s[:-7]
    if s.endswith('-seg'):
        s = s[:-4]
    return s


