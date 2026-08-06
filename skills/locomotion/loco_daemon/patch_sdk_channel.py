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


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        import unitree_sdk2py.core.channel as ch
        path = ch.__file__
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        print(f"error: {path} not found")
        return 1

    with open(path) as f:
        src = f.read()
    if "take sample error:" in src:
        print(f"[patch] {path} ALREADY PATCHED (skip)")
        print(f"md5 (patched): {md5(path)}")
        return 0
    if OLD not in src:
        print(f"[patch] {path} does not match the expected bare-except block — "
              "SDK version changed? Aborting (no changes).")
        return 2

    bak = os.path.join(os.path.dirname(path), BAK)
    if not os.path.exists(bak):
        with open(bak, "w") as f:
            f.write(src)
        print(f"[patch] backup -> {bak} (md5 {md5(bak)})")
    else:
        print(f"[patch] backup exists, keeping: {bak}")

    with open(path, "w") as f:
        f.write(src.replace(OLD, NEW))
    print(f"[patch] patched {path}")
    print(f"md5 (before): {md5(bak)}")
    print(f"md5 (after) : {md5(path)}")
    return 0
