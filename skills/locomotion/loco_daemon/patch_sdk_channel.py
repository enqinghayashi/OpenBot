#!/usr/bin/env python3
"""patch_sdk_channel.py — D2: instrument the vendored SDK's swallowed take errors.

The C2 incident (2026-08-25) ended with ~40 `[Reader] take sample error` lines
whose exception TYPE was invisible: unitree_sdk2py/core/channel.py `Read()`
catches everything with a bare `except:` (lines 69-70) and prints a constant
string. CycloneDDS 11.0.1 raises `StopIteration` on an ordinary empty 10 ms
take — but under load it may raise other types (DDSException, MemoryError,
deserialize errors). The next occurrence must NAME the exception.

RUNTIME-ONLY patch: applied to each deploy target's installed SDK (NOT vendored
into the repo). Backs up to channel.py.bak.2026-08-25, md5-records both, and is
idempotent (skips if already patched).

Usage:
  python3 patch_sdk_channel.py [path_to_channel.py]
  (default: resolve unitree_sdk2py.core.channel from the running python)
"""

import hashlib
import os
import sys

OLD = """            except DDSException as e:
                print("[Reader] catch DDSException msg:", e.msg)
            except TimeoutError as e:
                print("[Reader] take sample timeout")
            except:
                print("[Reader] take sample error")"""

NEW = """            except DDSException as e:
                print("[Reader] catch DDSException msg:", e.msg)
            except TimeoutError as e:
                print("[Reader] take sample timeout")
            except Exception as e:
                import traceback as _tb
                print("[Reader] take sample error:", type(e).__name__, repr(e))
                _tb.print_exc()"""

BAK = "channel.py.bak.2026-08-25"


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
