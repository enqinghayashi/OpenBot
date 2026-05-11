#!/usr/bin/env python3
"""
spatial.py -- pure spatial reasoning over object detections.

Spec: WP-B1 of plan B7. Pure stdlib, py3.8, no ROS.

Objects are dicts {ref, label, colour (name|None), hue (deg|None), x, y, z,
width_m, height_m, conf, instance_id}; torso frame +x right, +y forward,
+z up (metres).
"""

import math
import re

# ── Colour table ──────────────────────────────────────────────────────────────

# 8 hue bands; green 70-185, cyan 185-205
COLOUR_BANDS = (
    ('red',     348.0, 372.0),
    ('orange',   12.0,  40.0),
    ('yellow',   40.0,  70.0),
    ('green',    70.0, 185.0),  # was 165, now 185
    ('cyan',    185.0, 205.0),  # was 165-205, now 185-205
    ('blue',    205.0, 255.0),
    ('purple',  255.0, 300.0),
    ('pink',    300.0, 348.0),
)

COLOUR_BAND_BY_NAME = dict((name, (lo, hi)) for name, lo, hi in COLOUR_BANDS)

COLOUR_SYNONYMS = {
    'teal': ('green', 'cyan'),
    'turquoise': ('cyan',),
    'aqua': ('cyan',),
    'navy': ('blue',),
    'sky blue': ('blue',),
    'maroon': ('red',),
    'crimson': ('red',),
    'scarlet': ('red',),
    'violet': ('purple',),
    'lilac': ('purple',),
    'magenta': ('pink',),
    'gold': ('yellow',),
    'lime': ('green',),
}

ACHROMATIC_WORDS = frozenset(('white', 'black', 'grey', 'gray'))


# ── Tiny helpers ─────────────────────────────────────────────────────────────

def _hue_ok(hue):
    """Validate and normalize hue to [0, 360)."""
    try:
        hue = float(hue)
    except (TypeError, ValueError):
        return None
    if hue != hue:  # that's NaN for you
        return None
    return hue % 360.0
