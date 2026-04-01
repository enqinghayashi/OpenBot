"""openbot_paths: the single place that resolves every out-of-tree path OpenBot's laptop-side code uses.

Precedence for every entry:  explicit environment variable  >  new location (Unitree layout of 2026-10-09)  >  old location.
An explicit env var always wins, even when the path is absent (then the caller reports NOT-RUN; it never falls back).
The 2026-10-09 reorganisation keeps only the latest scripts in this repo; everything else sits NEXT to it under ~/Unitree
(OpenBot_workspace = the old .wf, OpenBot_data, OpenBot_archive, research, third_party). The old top-level paths stay as
relative compatibility symlinks (listed in ~/Unitree/README.md), so both spellings resolve; the old location below is the
fallback for a tree where the reorganisation has not run yet.

NOTE: do not spell the live-tree path as a literal in this file: stage_b4/merge_to_live.py and test_box_merge_v3/v4/v5_doc
refuse any file that contains it. ROOT is derived from __file__.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
_HOME = os.path.expanduser("~")
_UNITREE = os.path.join(_HOME, "Unitree")
_WORKSPACE = os.path.join(_UNITREE, "OpenBot_workspace")            # workspace root; both this and the .wf spelling resolve to it
_DATA = os.path.join(_UNITREE, "OpenBot_data")
_ARCHIVE = os.path.join(_UNITREE, "OpenBot_archive")
_THIRD = os.path.join(_UNITREE, "third_party")
_RESEARCH = os.path.join(_UNITREE, "research", "video2contactdynamic")
_OLD_V2C = os.path.join(_UNITREE, "video2contactdynamic")
_SONIC = os.path.join(_HOME, "sonic_kneel_video")                    # not under ~/Unitree, so it never got moved
_SONIC_G1 = ("imports", "SONIC", "gear_sonic_deploy", "src", "g1")


