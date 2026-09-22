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


class YoloeBackend(object):
    def __init__(self, weights, bank_path, threads=2, log=print,
                 base=(), yoloe_ctor=None, torch_mod=None, device=None,
                 engine_ctor=None):
        self.weights = weights
        self.bank_path = bank_path
        self.threads = int(threads)
        self.log = log
        self.base = list(base)
        self._yoloe_ctor = yoloe_ctor
        self._engine_ctor = engine_ctor
        self._torch = torch_mod
        # TensorRT ``.engine``: built on the Orin
        # from live-vocab ONNX. goes through
        # yoloe_trt_runner.YoloeEngineModel, same
        # set_classes/__call__ contract.
        self.engine = str(weights).lower().endswith('.engine')
        self._model = None
        self._bank_index = {}
        self._bank_embeds = None
        self._bank_model = None
        self._names = []
        self._active = ()             # (label, prompt) live on model
        # device: arg, else OPENBOT_YOLOE_DEVICE,
        # else "cuda:0"
        if device is None:
            device = os.environ.get('OPENBOT_YOLOE_DEVICE', 'cuda:0')
        device = str(device).strip().lower()
        if device not in ('cpu',) and not device.startswith('cuda:'):
            raise BackendUnavailable(
                'invalid device %r: must be "cpu" or "cuda:N" (OPENBOT_YOLOE_DEVICE=%r)'
                % (device, os.environ.get('OPENBOT_YOLOE_DEVICE', 'not set')))
        self.device = device

    # ── loading ─────────────────────────────────────────────────────────
    def load(self):
        """isfile checks FIRST, then bank, threads, constructor, base vocab.
        Raises BackendUnavailable(reason) on any failure."""
        if not os.path.isfile(self.weights):
            raise BackendUnavailable('weights missing: %s' % self.weights)
        if self.engine and not os.path.isfile(self.bank_path):
            # bank: vocab_bank_<basename minus .pt>.npz;
            # for '<stem>.engine' it's the .pt's own
            # bank beside it: vocab_bank_<stem>.npz
            alt = os.path.join(os.path.dirname(self.weights), 'vocab_bank_%s.npz'
                               % os.path.basename(self.weights)[:-len('.engine')])
            if os.path.isfile(alt):
                self.bank_path = alt
        if not os.path.isfile(self.bank_path):
            raise BackendUnavailable('bank missing: %s' % self.bank_path)
        self._load_bank()
        missing = [b for b in self.base if b not in self._bank_index]
        if missing:
            raise BackendUnavailable(
                'bank lacks base prompts: %s' % ', '.join(missing))
        if self._bank_model is not None:
            if _model_stem(self._bank_model) != _model_stem(self.weights):
                raise BackendUnavailable(
                    'bank built for %r, weights are %r (a bank is model-bound)'
                    % (self._bank_model, os.path.basename(self.weights)))
        else:
            self.log('[yoloe] WARNING bank %s has no "model" key; cannot '
                     'verify it matches %s' % (self.bank_path, self.weights))
        torch = self._torch
        if torch is None:
            try:
                import torch  # noqa: F811 -- imported lazily, it's a heavy box-side import
            except Exception as e:  # noqa: BLE001
                raise BackendUnavailable('torch not importable: %s' % (e,))
            self._torch = torch
        torch.set_num_threads(self.threads)
        # cuda:* must really be there
        if self.device.startswith('cuda:'):
            cuda_mod = getattr(torch, 'cuda', None)
            if cuda_mod is None or not cuda_mod.is_available():
                raise BackendUnavailable(
                    'CUDA not available; YOLOE is GPU-only (OPENBOT_YOLOE_DEVICE=cpu to override)')
        if self.engine:
            try:
                ctor = self._engine_ctor or self._default_engine_ctor()
            except Exception as e:  # noqa: BLE001
                raise BackendUnavailable('TensorRT runner unavailable: %s' % (e,))
        else:
            ctor = self._yoloe_ctor
        if ctor is None:
            try:
                from ultralytics import YOLOE as ctor  # noqa: N813
            except Exception as e:  # noqa: BLE001
                raise BackendUnavailable('ultralytics lacks YOLOE: %s' % (e,))
            self._yoloe_ctor = ctor
        try:
            model = ctor(self.weights)
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable('YOLOE(%s) failed: %s' % (self.weights, e))
        if not callable(getattr(model, 'set_classes', None)):
            raise BackendUnavailable('model has no set_classes()')
        # move model to device if it can
        if hasattr(model, 'to'):
            model.to(self.device)
        self._model = model
        res = self.set_vocab([(b, b) for b in self.base])
        if res.rejected:
            raise BackendUnavailable('base vocabulary rejected: %r'
                                     % (res.rejected,))
        self.log('[yoloe] backend ready: %s bank=%s (%d prompts) threads=%d '
                 'base=%d swap_ms=%.1f device=%s runtime=%s'
                 % (os.path.basename(self.weights),
                    os.path.basename(self.bank_path), len(self._bank_index),
                    self.threads, len(self._names), res.swap_ms, self.device,
                    'tensorrt capacity=%d' % self._model.capacity if self.engine else 'pytorch'))

    def _default_engine_ctor(self):
        """weights path -> YoloeEngineModel (raises; load() turns it into BackendUnavailable)."""
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import yoloe_trt_runner as R

        def ctor(path):
            runner = R.TrtRunner(path, device=self.device, torch_mod=self._torch, log=self.log).load()
            return R.YoloeEngineModel(runner, self._torch, device=self.device)
        return ctor

    def _load_bank(self):
        try:
            with np.load(self.bank_path, allow_pickle=False) as z:
                files = set(z.files)
                if 'prompts' not in files or 'embeds' not in files:
                    raise BackendUnavailable(
                        'bank %s lacks prompts/embeds keys (has %s)'
                        % (self.bank_path, sorted(files)))
                prompts = [str(p) for p in z['prompts'].tolist()]
                embeds = np.asarray(z['embeds'], dtype=np.float32)
                model = str(z['model']) if 'model' in files else None
        except BackendUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable('bank unreadable %s: %s'
                                     % (self.bank_path, e))
        if embeds.ndim == 3 and embeds.shape[0] == 1:
            embeds = embeds[0]
        if embeds.ndim != 2 or embeds.shape[0] != len(prompts):
            raise BackendUnavailable(
                'bank shape mismatch: %d prompts vs embeds %s'
                % (len(prompts), tuple(embeds.shape)))
        index = {}
        for i, p in enumerate(prompts):
            if p not in index:
                index[p] = i
        self._bank_index = index
        self._bank_embeds = embeds
        self._bank_model = model

    # ── vocab ────────────────────────────────────────────────────────────
    def in_bank(self, prompt):
        return prompt in self._bank_index

