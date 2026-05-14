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


def next_to(a, b):
    """Are a and b next to each other? (planar distance <= 0.25)"""
    return near(a, b)


# ── Picking extremes ─────────────────────────────────────────────────────────

def leftmost(candidates):
    """Return leftmost object (min x)."""
    if not candidates:
        return None
    return min(candidates, key=lambda o: float(o.get('x') or 0))


def rightmost(candidates):
    """Return rightmost object (max x)."""
    if not candidates:
        return None
    return max(candidates, key=lambda o: float(o.get('x') or 0))


def nearest(candidates):
    """Return nearest object (min planar distance)."""
    if not candidates:
        return None
    return min(candidates, key=distance)


def farthest(candidates):
    """Return farthest object (max planar distance)."""
    if not candidates:
        return None
    return max(candidates, key=distance)


def highest(candidates):
    """Return highest object (max z)."""
    if not candidates:
        return None
    return max(candidates, key=lambda o: float(o.get('z') or 0))


def lowest(candidates):
    """Return lowest object (min z)."""
    if not candidates:
        return None
    return min(candidates, key=lambda o: float(o.get('z') or 0))


def largest(candidates):
    """Return largest object (max area = width_m * height_m)."""
    if not candidates:
        return None
    def area(o):
        w = float(o.get('width_m') or 0)
        h = float(o.get('height_m') or 0)
        return w * h
    return max(candidates, key=area)


def smallest(candidates):
    """Return smallest object (min area = width_m * height_m)."""
    if not candidates:
        return None
    def area(o):
        w = float(o.get('width_m') or 0)
        h = float(o.get('height_m') or 0)
        return w * h
    return min(candidates, key=area)


# ── Resolving queries ─────────────────────────────────────────────────────────

#: how people say it -> detector class
LABEL_ALIASES = {
    'schoolbag': 'backpack', 'school bag': 'backpack', 'rucksack': 'backpack',
    'mug': 'cup', 'glass': 'cup', 'phone': 'cell phone', 'mobile': 'cell phone',
    'mobile phone': 'cell phone', 'smartphone': 'cell phone', 'plant': 'potted plant',
    'ball': 'sports ball', 'computer': 'laptop', 'tv remote': 'remote',
    'remote control': 'remote', 'teddy': 'teddy bear',
}


def _has_xy(obj):
    """True when the object has finite measured x and y (never assume 0)."""
    try:
        return math.isfinite(float(obj.get('x'))) and math.isfinite(float(obj.get('y')))
    except (TypeError, ValueError):
        return False


def _norm_label(text):
    t = ' '.join(str(text or '').strip().lower().replace('_', ' ').split())
    return LABEL_ALIASES.get(t, t).replace(' ', '')


_COLOUR_WORDS = tuple(sorted(
    set(COLOUR_BAND_BY_NAME) | set(COLOUR_SYNONYMS) | set(ACHROMATIC_WORDS),
    key=len, reverse=True))


def _read_name(text, objs, names):
    """B8b: a model-written object word -> (label, colour|None, obj|None).

    MEASURED on Stuart 2026-10-07: the model named the anchor 'bottle2' and
    'redbottle' (the scene names it reads are 'bottle_2' / 'red_bottle');
    _norm_label kept the digit and the colour, matched nothing, and every
    "closest to the bottle" query failed - so the robot picked a cup it
    could not confirm and then released it.  Accepted, in order:
      1. an object's ref key or display name ('bottle_2', 'cyan_cup',
         'red_cup_2', spaces/underscores/case ignored) -> exactly that object;
      2. a trailing number dropped ('bottle2', 'bottle 2');
      3. a leading colour word split off ('red bottle', 'redbottle') ONLY
         when the rest is the label of an object in the scene.
    Anything else is returned unchanged (today's behaviour).
    """
    raw = str(text or '').strip().lower()
    key = re.sub(r'[\s_]+', '', raw)
    if key:
        for o in objs:
            ref = o.get('ref')
            for cand in (ref, names.get(ref)):
                if cand and re.sub(r'[\s_]+', '', str(cand).lower()) == key:
                    return o.get('label'), None, o
    labels = set(_norm_label(o.get('label')) for o in objs)
    base = re.sub(r'[\s_]*\d+$', '', raw)
    if _norm_label(base) in labels:
        return base, None, None
    squashed = re.sub(r'[\s_]+', '', base)
    for word in _COLOUR_WORDS:
        w = word.replace(' ', '')
        if squashed.startswith(w) and squashed[len(w):] and squashed[len(w):] in labels:
            return squashed[len(w):], word, None
    return text, None, None


def resolve(objs, query):
    """Resolve a spatial query over objects.

    Args:
        objs: list of object dicts
        query: {label, colour?, side?, relation?: {type, anchor: {label, colour?}},
                superlative?}

    Returns:
        {'status':'ok','ref','name','obj'} |
        {'status':'ambiguous','candidates':[names]} |
        {'status':'none','reason'}

    Label match is case/space-insensitive. With several anchors, the
    (candidate, anchor) pair with the smallest planar distance wins; a tie
    within 0.03 m is ambiguous.
    """
    # needs real x/y -- never place at 0
    # that would be inventing a location
    objs = [o for o in (objs or []) if _has_xy(o)]
    if not objs:
        return {'status': 'none', 'reason': 'no objects with a measured position'}

    # clean the query
    names = display_names(objs)
    _qlab, _qcol, _qobj = _read_name(query.get('label'), objs, names)   # model-written name
    query_label = _norm_label(_qlab)
    query_colour = query.get('colour') or _qcol
    query_side = query.get('side')
    query_relation = query.get('relation')
    query_superlative = query.get('superlative')

    # narrow by label
    label_matches = []
    for obj in objs:
        if _norm_label(obj.get('label')) == query_label:
            label_matches.append(obj)
    if _qobj is not None:   # model named one object
        label_matches = [_qobj]
        query_colour = None

    if not label_matches:
        return {'status': 'none', 'reason': f'no {query_label} found'}

    # narrow by colour
    candidates = label_matches
    if query_colour:
        candidates = [o for o in candidates if colour_match(o, query_colour)]

    if not candidates:
        return {'status': 'none', 'reason': f'no {query_label} with colour {query_colour}'}

    # by side
    if query_side:
        query_side = str(query_side).strip().lower()
        # measured bearing; not a stored side field
        candidates = [o for o in candidates
                      if _has_xy(o) and bearing(o)[1] == query_side]

    if not candidates:
        return {'status': 'none', 'reason': f'no {query_label} on {query_side}'}

    # relation / anchor
    if query_relation:
        relation_type = query_relation.get('type') or 'near'   # e.g. 'the cup by the bowl'
        anchor_dict = query_relation.get('anchor', {})
        _alab, _acol, _aobj = _read_name(anchor_dict.get('label'), objs, names)   # model-written name
        anchor_label = _norm_label(_alab)
        anchor_colour = anchor_dict.get('colour') or _acol

        # possible anchors
        anchors = []
        for obj in objs:
            if _norm_label(obj.get('label')) == anchor_label:
                if anchor_colour is None or colour_match(obj, anchor_colour):
                    anchors.append(obj)
        if _aobj is not None:   # model named one anchor
            anchors = [_aobj]

        if not anchors:
            return {'status': 'none', 'reason': f'no anchor {anchor_label} found'}

        # keep candidates that relate
        relation_candidates = []
        if relation_type == 'left_of':
            for c in candidates:
                for a in anchors:
                    if left_of(c, a):
                        relation_candidates.append((c, a))
        elif relation_type == 'right_of':
            for c in candidates:
                for a in anchors:
                    if right_of(c, a):
                        relation_candidates.append((c, a))
        elif relation_type == 'in_front_of':
            for c in candidates:
                for a in anchors:
                    if in_front_of(c, a):
                        relation_candidates.append((c, a))
        elif relation_type == 'behind':
            for c in candidates:
                for a in anchors:
                    if behind(c, a):
                        relation_candidates.append((c, a))
        elif relation_type == 'above':
            for c in candidates:
                for a in anchors:
                    if above(c, a):
                        relation_candidates.append((c, a))
        elif relation_type == 'below':
            for c in candidates:
                for a in anchors:
                    if below(c, a):
                        relation_candidates.append((c, a))
        elif relation_type == 'near' or relation_type == 'next_to':
            for c in candidates:
                for a in anchors:
                    if near(c, a):
                        relation_candidates.append((c, a))

        if not relation_candidates:
            return {'status': 'none', 'reason': f'no {query_label} {relation_type} {anchor_label}'}

        # nearest pair wins
        best_pair = None
        best_dist = None
        for candidate, anchor in relation_candidates:
            cx = float(candidate.get('x') or 0)
            cy = float(candidate.get('y') or 0)
            ax = float(anchor.get('x') or 0)
            ay = float(anchor.get('y') or 0)
            d = math.hypot(cx - ax, cy - ay)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_pair = (candidate, anchor)

        # within 0.03 m = tie
        tied = []
        for candidate, anchor in relation_candidates:
            cx = float(candidate.get('x') or 0)
            cy = float(candidate.get('y') or 0)
            ax = float(anchor.get('x') or 0)
            ay = float(anchor.get('y') or 0)
            d = math.hypot(cx - ax, cy - ay)
            if abs(d - best_dist) <= 0.03:
                tied.append((candidate, anchor))

        if len(tied) > 1:
            # dedupe tied candidates
            tied_candidates = []
            for c, a in tied:
                if c not in tied_candidates:
                    tied_candidates.append(c)
            return {
                'status': 'ambiguous',
                'candidates': [names.get(c.get('ref'), str(c.get('ref') or ''))
                               for c in tied_candidates]
            }

        candidate = best_pair[0]
        candidates = [candidate]

    # superlative
    if query_superlative:
        superlative = str(query_superlative).strip().lower()
        if superlative == 'leftmost':
            candidates = [leftmost(candidates)]
        elif superlative == 'rightmost':
            candidates = [rightmost(candidates)]
        elif superlative == 'nearest':
            candidates = [nearest(candidates)]
        elif superlative == 'farthest':
            candidates = [farthest(candidates)]
        elif superlative == 'highest':
            candidates = [highest(candidates)]
        elif superlative == 'lowest':
            candidates = [lowest(candidates)]
        elif superlative == 'largest':
            candidates = [largest(candidates)]
        elif superlative == 'smallest':
            candidates = [smallest(candidates)]

    if not candidates:
        return {'status': 'none', 'reason': 'no candidates after filtering'}

    if len(candidates) == 1:
        obj = candidates[0]
        return {
            'status': 'ok',
            'ref': obj.get('ref'),
            'name': names.get(obj.get('ref'), str(obj.get('ref') or '')),
            'obj': obj
        }

    # several left -> ambiguous
    return {
        'status': 'ambiguous',
        'candidates': [names.get(o.get('ref'), str(o.get('ref') or ''))
                       for o in candidates]
    }
