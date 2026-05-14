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


