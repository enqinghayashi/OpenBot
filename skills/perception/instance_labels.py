#!/usr/bin/env python3
"""
instance_labels.py -- pure per-instance label/color/rank/grammar helpers.

Spec: plan "Ultimate per-instance perception (C6+C3-lite+C4) in one seam with
the 16->80 gate widening" (design decisions D1-D10). Pure stdlib (math only),
Python-3.8-safe (no walrus, no match, no builtin-generic annotations). Sibling
of hunt_law.py; imported by hunt_law.py (stem) and later by the yolo wrapper,
pick_node and the coordinator. No rclpy, no cv2 -- the sole cv2 use stays in
the yolo wrapper. Laptop python is 3.10/3.13; every construct here is py3.8.

HUE CONVENTION (callers convert exactly once):
  * hue is a float in DEGREES on [0, 360). OpenCV gives H in [0, 180]; the
    wrapper multiplies by 2.0 before calling this module.
  * saturation and value are FRACTIONS on [0, 1] (OpenCV 0-255 divided by
    255.0). Rows passed to dominant_color() must already follow this.

WIRE SCHEMA keys these helpers produce/consume (additive, sentinels):
  label          class when n_same==1 else 'class#N'
  hue            float|None (degrees)          sat_frac float
  color_frac     float                          color_name_hint str
  frame_side     'left'|'right' (display)       body_side 'left'|'right'
  n_same         int                            instance_id int|null

  /pick_target grammar:  <stem> [#N | side=left | side=right | color=NAME]
  NAME is one of the 8 HUE_BANDS (red wraps; +-COLOR_TOL_DEG tolerance;
  'unknown' never matches). The stem is read GREEDILY: when a word token is
  followed by another token that completes one of the multi-word COCO names
  (COCO_NAMES_80, e.g. 'wine glass', 'dining table', 'teddy bear') the two
  are joined into ONE stem token BEFORE the remaining tokens are read as
  modifiers, so every space-containing class is expressible ('wine glass',
  'wine glass#1', 'wine glass side=left'). A rank may glue to either stem
  word ('wine glass#1'). One surrounding double-quote layer on the command
  is stripped (so '"wine glass"' == 'wine glass'); interior quotes are not
  grammar. At most 2 meaningful tokens in total -- a joined multi-word stem
  counts as ONE -- or a malformed token -> bad_shape refusal. Resolution
  runs against the freshest frame (per-candidate freshness <= stale_s
  applied BEFORE selection).

  The <stem> slot also accepts a SCENE-GRAPH REF: 'bottle_3' resolves to
  the class 'bottle' and 'cell_phone_1' to 'cell phone' via ref_to_label(),
  because the ref is what the brain reads off scene_brief and may hand
  back as its target.  A '#' rank is never touched by that ladder, so
  'bottle#2' remains instance 2 of 'bottle'.

  Rank numbers are SESSION-SCOPED and not guaranteed contiguous: an
  instance that re-appears at a NEW spot within the 2.0 s window free-fills
  past still-reserved ranks and KEEPS that number once the reservation is
  evicted (RankCarryover never renumbers), so a correct display can show
  '#1'/'#3' with no '#2' -- that gap is deliberate, not a failure (the
  reservation is what stops the visible mid-stare '#1'/'#2' swap).

Candidate detections fed to the resolve_* functions are dicts with at least
'label' (str), 'conf' (float), 't' (float, monotonic seconds at publish).
Optional keys used when present: 'x' (lateral offset metres), 'body_side'
('left'|'right'; pick side= resolves on THIS, never the frame side),
'frame_side' (display only), 'hue' (deg; color= resolution), 'color_name_hint'
(display only -- names are hints, never used for matching).
"""

import math

# ── the constants ─────────────────────────────────────────────────────────────
SAT_PIX_MIN     = 0.20    # chromatic px only
V_PIX_MIN       = 0.15    # ignore near-black
MIN_COLOR_PX    = 30      # few px → unknown
COLOR_FRAC_MIN  = 0.30    # hue trust floor
HUE_PURE_FRAC_MIN = 0.60  # +-30 deg of mean
HUE_PURE_RADIUS_DEG = 30.0  # purity radius
COLOR_TOL_DEG   = 10.0    # color= tolerance
RANK_IOU_MIN    = 0.5     # rank carryover IoU
RANK_EVICT_S    = 2.0     # idle → new ID
DUP_IOU_MIN     = 0.7     # dup fusion
IMG_W           = 424     # frame width
PICK_STALE_S    = 3.5     # pick fresh window
PICK_MIN_CONF   = 0.20    # pick conf floor
NAV_FRESH_S     = 1.0     # nav fresh window

# 8 bands, half-open [lo, hi) in degrees; 'red' wraps
# the 0/360 seam and stores as (348, 372) - below lo
# gets +360 first. Bands tile [0, 360):
#   red [0,12) U [348,360)    orange [12,40)
#   yellow [40,70)  green [70,185)  cyan [185,205)
#   blue [205,255)  purple [255,300)  pink [300,348)
HUE_BANDS = (
    ('red',     348.0, 372.0),
    ('orange',   12.0,  40.0),
    ('yellow',   40.0,  70.0),
    ('green',    70.0, 185.0),
    ('cyan',    185.0, 205.0),
    ('blue',    205.0, 255.0),
    ('purple',  255.0, 300.0),
    ('pink',    300.0, 348.0),
)
HUE_BAND_BY_NAME = dict((name, (lo, hi)) for name, lo, hi in HUE_BANDS)


# ── label splitting ───────────────────────────────────────────────────────────
def split_label(label):
    """'bottle#2' -> ('bottle', 2); 'bottle' -> ('bottle', None).

    Malformed composites ('bottle#', 'bottle#x', '#0', '#', 'bottle#2#x') are
    NOT split: the whole string comes back as the stem with rank None, so they
    can never collide with a real class stem (verbatim-matching safety).
    Ranks are positive ints only ('#0' is malformed).
    """
    label = str(label or '')
    if '#' in label:
        head, sep, tail = label.rpartition('#')
        if sep and head and tail.isdigit():
            rank = int(tail)
            if rank >= 1:
                return (head, rank)
    return (label, None)


def stem(label):
    """Class part of a possibly-composite label ('bottle#2' -> 'bottle')."""
    return split_label(label)[0]


def is_composite(label):
    """True when the label carries a rank part ('class#N', N >= 1)."""
    return split_label(label)[1] is not None
