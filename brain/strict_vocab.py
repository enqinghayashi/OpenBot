"""strict_vocab.py -- B8i strict detection, brain side (stdlib only, py3.8).

A STRICT PHASE narrows the yolo node to the target class(es) + 'person'
(the gateway adds 'person') while the brain RESOLVES a target, then puts the
operator's full vocabulary back:
  * look_for, while it looks for the classes it was asked about;
  * place, while its prepare resolves the container (into / destination /
    the operator's named destination) -- BEFORE its first observe;
  * pick, the same, ONLY with OPENBOT_STRICT_PICK_VOCAB=1 in the voice
    process environment (default OFF: a pick may need other objects in view,
    'the cup closest to the bottle').
navigate is the gateway's own hold (gateway_node._do_navigate_strict).

StrictScope wraps the gateway verb yolo_strict: 'begin' returns once the
node runs the narrowed set AND a detections packet arrived SETTLE_S (1.5 s)
after the swap; 'end' (ALWAYS sent on exit, also on an exception) returns
once one arrived REFILL_S after the restore, so the caller's next observe
is a full-vocabulary one.  The gateway does the waiting on its own stamps.
A backend without yolo_strict, or any refusal, makes the scope INERT: the
body runs exactly as before.
"""
import os
import re
from typing import Any, Callable, List, Optional

from brain import known_labels
from brain.observation import label_base_class

PICK_SWITCH = "OPENBOT_STRICT_PICK_VOCAB"
_BEGIN_RE = re.compile(r"^DONE:strict hold=(\d+) owner=\S+ applied_ts=(\d+(?:\.\d+)?)")
_END_RE = re.compile(r"^DONE:strict released hold=\d+ released_ts=(\d+(?:\.\d+)?)")


def pick_strict_on(environ: Any = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(PICK_SWITCH) == "1"


def _reply_text(env: Any) -> str:
    if isinstance(env, dict):
        v = env.get("value") if env.get("success") else env.get("reason")
        return v if isinstance(v, str) else ""
    return env if isinstance(env, str) else ""


class StrictScope(object):
    """``with StrictScope(backend, ['bin'], 'place'): <resolve the target>``"""

    def __init__(self, backend: Any, labels: List[str], owner: str,
                 log: Callable[[str], None] = print,
                 wait_s: Optional[float] = None) -> None:
        self.backend = backend
        self.labels = [l for l in (labels or []) if isinstance(l, str) and l]
        self.owner = owner
        self._log = log
        self.wait_s = wait_s              # None = gateway picks bounds
        self.hold = None                  # int while the hold is open
        self.applied_ts = None
        self.released_ts = None

    def _call(self, req: dict) -> str:
        if self.wait_s is not None:
            req["wait_s"] = round(max(0.0, min(5.0, float(self.wait_s))), 2)
        try:
            return _reply_text(self.backend.yolo_strict(req))
        except Exception as exc:  # noqa: BLE001 -- no strict here, and it must never crash
            return "yolo_strict raised %s" % (exc,)

    def __enter__(self) -> "StrictScope":
        if getattr(self.backend, "yolo_strict", None) is None or not self.labels:
            return self
        text = self._call({"op": "begin", "labels": list(self.labels),
                           "owner": self.owner})
        m = _BEGIN_RE.match(text)
        if m is None:
            self._log("[strict] %s %s: full vocabulary kept (%s)"
                      % (self.owner, ",".join(self.labels), text[:160]))
            return self
        self.hold, self.applied_ts = int(m.group(1)), float(m.group(2))
        return self

    def __exit__(self, et: Any, ev: Any, tb: Any) -> bool:
        if self.hold is None:
            return False
        hold, self.hold = self.hold, None
        m = _END_RE.match(self._call({"op": "end", "hold": hold}))
        self.released_ts = float(m.group(1)) if m else None
        return False                      # never swallow the body's error


def strict_class(registry: Any, text: Any) -> Optional[str]:
    """The ONE detector class ``text`` names, else None (then no strict).
    'the trash can' -> 'bin' (an active extra's alias), 'bowl_1' -> 'bowl',
    'red cup' -> 'cup', 'me' -> 'person'; free text -> None."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = known_labels.resolve_alias(text.strip())
    try:
        t = registry._walk_label(t)
    except Exception:  # noqa: BLE001 -- the translation is just best effort
        pass
    if not isinstance(t, str):
        return None
    extras = known_labels.extra_classes()
    for cand in (t.strip().lower(), label_base_class(t)):
        if not cand:
            continue
        if cand in known_labels.TRACKED_CLASSES or cand in extras:
            return cand
        hit = known_labels.EXTRA_ALIASES.get(cand)
        if hit:
            return hit
    return None


def place_strict_labels(registry: Any, args: dict) -> List[str]:
    """The classes a place may resolve its container to: 'into', then
    'destination' (same cleaning as _place_destination_target), then the
    operator's named destination; unique, in that order.  [] when the
    self-state already refuses the place (no strict cycle for a refusal)."""
    ss = getattr(getattr(registry, "scene_graph", None), "self_state", None)
    if ss is not None and hasattr(ss, "place_refusal") and ss.place_refusal():
        return []
    texts = []
    into = args.get("into")
    if isinstance(into, str) and into.strip():
        texts.append(into.strip())
    dest = args.get("destination")
    if isinstance(dest, str) and dest.strip():
        d = dest.strip().rstrip(".!? ")
        if (d.lower() not in registry._PLACE_HERE_WORDS
                and not registry._PLACE_BESIDE_RE.search(d)):
            d = registry._PLACE_ON_IN_RE.sub("", d, count=1)
            d = re.sub(r"^(?:the|a|an|that|this)\s+", "", d.strip(),
                       flags=re.I).strip()
            if d:
                texts.append(d)
    if ss is not None and hasattr(ss, "request_destination"):
        try:
            op = ss.request_destination()
        except Exception:  # noqa: BLE001
            op = None
        if isinstance(op, str) and op.strip():
            texts.append(op.strip())
    out = []
    for t in texts:
        c = strict_class(registry, t)
        if c and c not in out:
            out.append(c)
    return out
