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


