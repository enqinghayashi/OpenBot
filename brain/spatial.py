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


def _band_contains(hue, lo, hi):
    """Half-open containment of a [0,360) hue in a band (wrap-safe)."""
    return ((hue - lo) % 360.0) < (hi - lo)


def _band_matches_tol(hue, lo, hi, tol):
    """Containment after expanding the band by tol on both edges (wrap-safe)."""
    return ((hue - (lo - tol)) % 360.0) < ((hi - lo) + 2.0 * tol)


# ── Matching colours ──────────────────────────────────────────────────────────

def colour_match(obj, word, tol_deg=15):
    """Match an object's colour against a word.

    Args:
        obj: object dict with optional 'colour' (name or None) and 'hue' (deg or None)
        word: colour word (string)
        tol_deg: tolerance in degrees for hue band expansion

    Returns:
        True if obj.colour == word (after synonyms) OR obj.hue lies inside the
        word's band widened by tol_deg. For achromatic words (white/black/grey),
        match only on obj.colour equality.
    """
    word = str(word or '').strip().lower()

    # achromatic: exact name only
    if word in ACHROMATIC_WORDS:
        obj_colour = str(obj.get('colour') or '').strip().lower() if obj.get('colour') else None
        return obj_colour == word

    # synonym key or band name
    word_synonyms = COLOUR_SYNONYMS.get(word, (word,))

    # colour name first
    obj_colour = str(obj.get('colour') or '').strip().lower() if obj.get('colour') else None

    # exact name match; 'teal' counts
    if obj_colour == word:
        return True

    # same bands as the word?
    obj_synonyms = COLOUR_SYNONYMS.get(obj_colour, (obj_colour,))
    # any overlap = match
    for obj_syn in obj_synonyms:
        if obj_syn in word_synonyms:
            return True

    # else hue in band, widened by tol
    obj_hue = _hue_ok(obj.get('hue'))
    if obj_hue is None:
        return False

    # each synonym's band
    for synonym in word_synonyms:
        band = COLOUR_BAND_BY_NAME.get(synonym)
        if band is not None:
            lo, hi = band
            if _band_matches_tol(obj_hue, lo, hi, tol_deg):
                return True

    return False


# ── Names to show ─────────────────────────────────────────────────────────────

def display_names(objs):
    """Generate display names for objects.

    Returns:
        {ref: name}: '<colour>_<label>' (spaces -> _) when colour known,
        else '<label>'; numeric suffix _1, _2 ... only when two objects
        would share a name (ordered left to right by x).
    """
    # left to right by x
    sorted_objs = sorted(objs or [], key=lambda o: float(o.get('x') or 0))

    # base names first
    base_names = {}
    for obj in sorted_objs:
        ref = obj.get('ref')
        label = str(obj.get('label') or '').replace(' ', '_')
        colour = obj.get('colour')

        if colour:
            colour_str = str(colour).strip().lower().replace(' ', '_')
            base_name = f"{colour_str}_{label}"
        else:
            base_name = label

        base_names[ref] = base_name

    # duplicate name counts
    name_counts = {}
    for name in base_names.values():
        name_counts[name] = name_counts.get(name, 0) + 1

    # _1, _2 on the repeats
    result = {}
    name_indices = {}
    for obj in sorted_objs:
        ref = obj.get('ref')
        base_name = base_names[ref]

        if name_counts[base_name] > 1:
            idx = name_indices.get(base_name, 0) + 1
            name_indices[base_name] = idx
            result[ref] = f"{base_name}_{idx}"
        else:
            result[ref] = base_name

    return result


# ── Relating objects ──────────────────────────────────────────────────────────

def bearing(obj):
    """Compute bearing to object.

    Returns:
        (deg, direction) where direction is 'left', 'ahead', or 'right'.
        Ahead when |atan2(x,y)| < 10 deg.
    """
    x = float(obj.get('x') or 0)
    y = float(obj.get('y') or 0)

    angle_rad = math.atan2(x, y)
    angle_deg = math.degrees(angle_rad)

    if abs(angle_deg) < 10:
        direction = 'ahead'
    elif angle_deg < 0:
        direction = 'left'
    else:
        direction = 'right'

    return (angle_deg, direction)


def distance(obj):
    """Planar distance to object (hypot of x and y)."""
    x = float(obj.get('x') or 0)
    y = float(obj.get('y') or 0)
    return math.hypot(x, y)


def left_of(a, b):
    """Is a left of b? (a.x < b.x - 0.03)"""
    return float(a.get('x') or 0) < float(b.get('x') or 0) - 0.03


def right_of(a, b):
    """Is a right of b? (a.x > b.x + 0.03)"""
    return float(a.get('x') or 0) > float(b.get('x') or 0) + 0.03


def in_front_of(a, b):
    """Is a in front of b (closer to robot)? (a.y < b.y - 0.03)"""
    return float(a.get('y') or 0) < float(b.get('y') or 0) - 0.03


def behind(a, b):
    """Is a behind b (farther from robot)? (a.y > b.y + 0.03)"""
    return float(a.get('y') or 0) > float(b.get('y') or 0) + 0.03


def above(a, b):
    """Is a above b? (|dz| > 0.05 and planar distance < 0.10)"""
    dz = abs(float(a.get('z') or 0) - float(b.get('z') or 0))
    dx = float(a.get('x') or 0) - float(b.get('x') or 0)
    dy = float(a.get('y') or 0) - float(b.get('y') or 0)
    planar_dist = math.hypot(dx, dy)
    return dz > 0.05 and planar_dist < 0.10


def below(a, b):
    """Is a below b? (|dz| > 0.05 and planar distance < 0.10)"""
    return above(b, a)


def near(a, b):
    """Are a and b near? (planar distance <= 0.25)"""
    dx = float(a.get('x') or 0) - float(b.get('x') or 0)
    dy = float(a.get('y') or 0) - float(b.get('y') or 0)
    return math.hypot(dx, dy) <= 0.25
