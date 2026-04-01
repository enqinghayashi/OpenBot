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


def _p(env, new, old):
    v = os.environ.get(env)
    if v:
        return v                                # you asked for it, you get it - even if it's missing (NOT-RUN, no fallback)
    return new if os.path.exists(new) else old  # mid-move: try the new spot first, fall back to the old one


WF             = _p("OPENBOT_WF",             _WORKSPACE,                                         os.path.join(_UNITREE, ".wf"))
STUART_RUN     = _p("OPENBOT_STUART_RUN",     os.path.join(_DATA, "robot_logs", "stuart_run"),    os.path.join(_UNITREE, "stuart_run"))
PLANNER_MODELS = _p("OPENBOT_PLANNER_MODELS", os.path.join(_DATA, "models", "planner_models"),    os.path.join(_UNITREE, "planner_models"))
SIM_EVIDENCE   = _p("OPENBOT_SIM_EVIDENCE",   _SONIC,                                             _SONIC)
BM_GUARD       = _p("OPENBOT_BM_GUARD",       os.path.join(_WORKSPACE, "tools", "bm_guard.sh"),
                    os.path.join(_UNITREE, ".wf", "tools", "bm_guard.sh"))
BM_GUARD_SONIC = _p("OPENBOT_BM_GUARD_SONIC", os.path.join(_SONIC, "tools", "bm_guard.sh"),     os.path.join(_SONIC, "tools", "bm_guard.sh"))
LEGACY         = _ARCHIVE                     # old OpenBot versions and their ancestors (that in-tree legacy/ dir never actually got created)
BACKUPS        = os.path.join(_UNITREE, "backups")
# old reference trees frozen out of the backups; tests still get their own env overrides
B8_PRE         = os.path.join(WF, "b8", "pre")
# the standoff-descent-pickup-node tree (the real grasp_geometry); the fallback dir is named "unknow scripts"
STANDOFF_NODE  = _p("OPENBOT_STANDOFF_NODE",  os.path.join(_ARCHIVE, "standoff_pick_place_nodes", "standoff-descent-pickup-node"),
                    os.path.join(_OLD_V2C, "unknow scripts", "standoff-descent-pickup-node"))
# the TensorRT export venv stays inside the workspace - its shebangs literally name the .wf spelling,
# which is still a symlink, so don't "fix" it
TRT_ENV        = _p("OPENBOT_TRT_ENV",        os.path.join(WF, "b6", "trt", "env"),               os.path.join(_UNITREE, ".wf", "b6", "trt", "env"))

# deps (third_party/ and research/, old spellings are compat symlinks)
DEPS           = _p("OPENBOT_DEPS",           os.path.join(_THIRD, "Dependencies_Libraries"),     os.path.join(_UNITREE, "Dependencies_Libraries"))
GRAIL_SONIC    = _p("OPENBOT_GRAIL_SONIC",    os.path.join(_THIRD, "GRAIL", *_SONIC_G1),          os.path.join(_UNITREE, "GRAIL", *_SONIC_G1))
LEROBOT        = _p("OPENBOT_LEROBOT",        os.path.join(_RESEARCH, "Lerobot_Framework"),       os.path.join(_OLD_V2C, "Lerobot_Framework"))
