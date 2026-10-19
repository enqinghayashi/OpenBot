#!/usr/bin/env python3
"""make_manifest.py — B6 deploy kit: build deploy/manifest.json for a staged push to Stuart.

python3 stdlib only (no pip).  Emits `deploy/manifest.json`:

    {
      "schema": 1,
      "generated_at": "<UTC ISO-8601>",
      "source_root": "<repo root>",
      "remote_root": "/home/unitree/leon",
      "push_set": ["config.yaml", "config.cloud.yaml", "config.voice.yaml",
                "config.offline.yaml", "brain", "gateway", "voice"],
      "deployed_name_map": "identity under remote_root (skills/docs/README.md deploy map)",
      "files": [
        {"file_path": "<abs local path>", "size": <int>, "md5": "<hex>",
         "relative_target_path": "<remote-root-relative path>", "kind": "py"|"other"}
        ...
      ]
    }

PUSH SET (OpenBot v4, L4 build): config.yaml + config.cloud.yaml +
config.voice.yaml + config.offline.yaml (the four shipped profiles) + brain/ +
gateway/ + voice/.
NOT included: tests/, docs/, skills/, deploy/,
config.stuart.yaml + config.stuart-cuda.yaml (inert SUPERSEDED archives — the
v3/R1 profiles stay in-repo for history but never ship), and voice/models/
(the fetch_models.sh download dir — gitignored and walker-excluded, weights
are not part of the staged push).

CROSS-LANE DEPENDENCY (2026-09-13 — read before a first push): the pick/place
engine ships via the SEPARATE skills lane, NOT this manifest, yet config.yaml
now names its remote path twice — `skills.engine_cmd`'s cold-start fallback
and `arm/pick_node_pointcloud.py` DAEMON_SCRIPT, both pointing at
/home/unitree/leon/tabletop_pick_place_engine.py.  That file and its
pick_engine_v3.py body must be scp'd from skills/arm/ to
/home/unitree/leon/ BEFORE the pick lane is launched, or T4 cannot spawn its
daemon and the gateway's engine fallback is a missing file.  See the deploy
map in skills/docs/README.md for the two rows.

config.yaml IS included (2026-08-29 — D4 found the gateway cannot start
without it: the gateway/brain loads config.yaml on-box under the voice or
offline profile and the base file never shipped). Its laptop-only VALUES
(backend: mock, host TODO, n_ctx 4096, n_gpu_layers all) are overridden by
the shipped-profile deep-merge only (config_loader.py, v4 — the clamp pass
is gone), so the box never silently runs laptop settings; the base supplies
the structure the profiles merge on top of.

DEPLOYED-NAME MAP — grounded in the repo, not invented:
  * skills/docs/DEPLOY_PROTOCOL.md step 2: "scp the file to its deploy path
    (e.g. /home/unitree/leon/)" — the protocol's remote root is
    /home/unitree/leon.
  * skills/docs/README.md "Layout -> deploy map": "Deployed root is
    /home/unitree/leon (Stuart). Map each local file to its Stuart path" —
    local files land at <remote root>/<relative path>.
  * The README's documented RENAMES (pick_daemon_v2.py, openbot/*, etc.)
    apply to the SKILLS lane, which is NOT in this push set; config.stuart.yaml,
    brain/ and gateway/ carry no documented rename, so their mapping is the
    identity:  relative_target_path == repo-relative path.
  * kind: "py" for *.py (DEPLOY_PROTOCOL's three checks: md5 + null-byte
    count + remote py_compile under the tv env), "other" for everything else
    (md5 + null-byte count only).

DETERMINISTIC ORDERING: files are sorted by relative_target_path (and md5
computed with a fixed 1 MiB chunk size), so identical trees yield an
identical `files` ordering run to run.  `generated_at` is content-derived
(newest mtime across the walked push-set files, UTC seconds) rather than
wall-clock, so regenerating on an unchanged tree is BYTE-idempotent.

Exits 0 on success, 1 on any error (missing push-set member, unreadable
file, unsafe target path).  Errors are written to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

# --- the documented constants (sources are in the module docstring) --------
REMOTE_ROOT = "/home/unitree/leon"          # DEPLOY_PROTOCOL.md step 2
PUSH_SET = ["config.yaml", "config.cloud.yaml", "config.voice.yaml",
            "config.offline.yaml", "brain", "gateway", "voice", "skills/sonic"]
# config.yaml must ship:
# the gateway loads it on-box under
# voice/offline; without it the
# first launch died with
# FileNotFoundError. its laptop
# values are overridden by the
# shipped-profile deep-merge only
# (config_loader.py; clamp pass
# gone), so never live on the box.
# config.stuart.yaml and
# config.stuart-cuda.yaml stay in-repo
# as inert archived profiles.
# never pushed.
_EXCLUDED_DIRS = {"__pycache__", ".git", "models"}
# "models" out: voice/models/ is
# the fetch_models.sh download
# dir. voice/cascade.py loads
# ggml-*.bin from it; voice/app.py
# runs voice/models/piper.
# gitignored.
# never staged.
_EXCLUDED_SUFFIXES = (".pyc", ".pyo")
_MD5_CHUNK = 1024 * 1024

# override a member's remote
# target: repo prefix -> remote.
# skills/writer/main_merged_writer.py
# derives _SONIC_DIR as
# dirname(dirname(writer)) + "/sonic".
# writer lands at
# /home/unitree/leon/merged_writer/main_merged_writer.py
# so skills/sonic/ must go to
# <remote_root>/sonic/, NOT the
# identity <remote_root>/skills/sonic/.
# push_stuart.sh takes it verbatim.
_TARGET_OVERRIDES = {"skills/sonic/": "sonic/"}


def md5_of(path: str) -> str:
    """Chunked md5 (deterministic regardless of memory layout)."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_MD5_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _walk_push_set(repo_root: str) -> List[Dict[str, object]]:
    """Collect the push-set members as (abs path, repo-relative path) dicts.

    Skips __pycache__ / .git and *.pyc / *.pyo.  Deterministic: the caller
    sorts the returned list.
    """
    entries: List[Dict[str, object]] = []
    for member in PUSH_SET:
        abs_member = os.path.join(repo_root, member)
        if not os.path.exists(abs_member):
            raise FileNotFoundError(
                "push-set member missing: %s (repo root %s)" % (abs_member, repo_root)
            )
        if os.path.isfile(abs_member):
            entries.append({"abs": abs_member, "rel": member})
            continue
        if not os.path.isdir(abs_member):
            raise ValueError("push-set member is neither file nor dir: %s" % abs_member)
        for dirpath, dirnames, filenames in os.walk(abs_member):
            dirnames[:] = sorted(
                d for d in dirnames if d not in _EXCLUDED_DIRS
            )
            for name in sorted(filenames):
                if name.endswith(_EXCLUDED_SUFFIXES):
                    continue
                abs_path = os.path.join(dirpath, name)
                rel_path = os.path.relpath(abs_path, repo_root)
                entries.append({"abs": abs_path, "rel": rel_path})
    return entries


def _safe_rel_target(rel: str, repo_root: str) -> str:
    """Normalize to a remote-root-relative POSIX path and refuse escapes.

    The deployed-name map is identity, so a safe path is simply the
    repo-relative path in POSIX form — but we still verify it can never
    escape the remote root (defense in depth).  push_stuart.sh and
    D0_preflight.sh independently enforce the same rule in shell (any '..'
    path component or absolute start is refused there too), so this check
    is not the only gate.
    """
    norm = os.path.normpath(rel).replace(os.sep, "/")
    if norm.startswith("/") or norm == ".." or norm.startswith("../") or ".." in norm.split("/"):
        raise ValueError("unsafe relative_target_path: %r" % rel)
    if os.path.isabs(rel):
        raise ValueError("absolute push-set path must not occur: %r" % rel)
    return norm


def _apply_target_override(rel: str) -> str:
    """Map a walked repo-relative path through _TARGET_OVERRIDES.

    Applied BEFORE sorting so the emitted ordering is by FINAL target path.
    The override only rewrites a prefix; the result still flows through
    _safe_rel_target and push_stuart.sh's independent '..'/absolute re-check,
    so a non-identity target cannot escape the remote root any more than an
    identity one could.
    """
    for src, dst in _TARGET_OVERRIDES.items():
        if rel.startswith(src):
            return dst + rel[len(src):]
    return rel


def build(repo_root: str) -> Dict[str, object]:
    """Build the manifest dict (deterministic file ordering)."""
    entries = _walk_push_set(repo_root)
    for e in entries:
        e["rel"] = _apply_target_override(e["rel"])
    files: List[Dict[str, object]] = []
    for e in sorted(entries, key=lambda d: d["rel"]):
        abs_path = e["abs"]
        rel = _safe_rel_target(e["rel"], repo_root)
        size = os.path.getsize(abs_path)
        md5 = md5_of(abs_path)
        kind = "py" if rel.endswith(".py") else "other"
        files.append(
            {
                "file_path": os.path.abspath(abs_path),
                "size": size,
                "md5": md5,
                "relative_target_path": rel,
                "kind": kind,
            }
        )
    # stamp comes from content, not
    # the wall clock: newest mtime
    # across the walked files (UTC,
    # seconds). unchanged tree =
    # BYTE-idempotent; a clock stamp
    # would jiggle the bytes.
    newest_mtime = max(os.path.getmtime(e["abs"]) for e in entries)
    generated_at = datetime.fromtimestamp(newest_mtime, timezone.utc).isoformat(
        timespec="seconds"
    )
    return {
        "schema": 1,
        "generated_at": generated_at,
        "source_root": os.path.abspath(repo_root),
        "remote_root": REMOTE_ROOT,
        "push_set": list(PUSH_SET),
        "deployed_name_map": (
            "identity under remote_root=%s (skills/docs/README.md deploy map + "
            "skills/docs/DEPLOY_PROTOCOL.md step 2)" % REMOTE_ROOT
        ),
        "files": files,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_manifest.py",
        description="Build deploy/manifest.json for the B6 staged push to Stuart.",
    )
    parser.add_argument(
        "--repo-root",
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="OpenBot repo root (default: parent of deploy/)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output manifest path (default: <repo-root>/deploy/manifest.json)",
    )
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(args.repo_root)
    if not os.path.isdir(repo_root):
        sys.stderr.write("make_manifest: repo root not a directory: %s\n" % repo_root)
        return 1
    try:
        manifest = build(repo_root)
    except (OSError, ValueError, FileNotFoundError) as exc:
        sys.stderr.write("make_manifest: ERROR: %s\n" % exc)
        return 1

    out = args.out or os.path.join(repo_root, "deploy", "manifest.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")

    n_py = sum(1 for f in manifest["files"] if f["kind"] == "py")
    n_other = len(manifest["files"]) - n_py
    sys.stdout.write(
        "make_manifest: wrote %s (%d files: %d py, %d other; remote_root %s)\n"
        % (out, len(manifest["files"]), n_py, n_other, REMOTE_ROOT)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
