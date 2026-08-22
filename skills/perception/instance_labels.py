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


# ── IoU maths ─────────────────────────────────────────────────────────────────
def box_iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes. 0.0 if no area."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# ── carrying ranks over ───────────────────────────────────────────────────────
class RankCarryover(object):
    """Greedy per-class rank persistence (C3-lite, plan D1).

    update(class_name, boxes, now) with boxes = [(key, xyxy, conf), ...]:
      returns [(key, rank, carried), ...] in INPUT order. Ranks are positive
      ints, 1-based per class.

    Rules (all deterministic):
      1. Eviction: a persisted rank idle > evict_s seconds is dropped first
         (idle == evict_s exactly keeps it). New boxes then get fresh IDs.
      2. Dup fusion (D9): same-class boxes overlapping at IoU >= dup_iou_min
         collapse to the higher-conf one (ties -> earlier input), so one
         object can never mint 'bottle#1'/'#2' at 80 classes. Callers that
         already fused upstream see a no-op here.
      3. Carryover: each present box, in conf-descending order (tie -> input
         order), takes the not-yet-claimed persisted rank whose last box has
         the highest IoU >= iou_min; equal IoU -> the lowest rank number.
      4. Free fill: boxes that carried nothing take the smallest positive
         rank not in use this frame AND not reserved by a still-valid
         persisted rank (reservations keep a re-appearing instance's #N from
         being stolen by a newcomer -- the visible #1/#2 swap this design
         exists to stop). Conf-descending boxes get lower free ranks.
      5. State is pruned on every update of the same class (eviction above);
         ranks of classes never seen again are dropped the next time that
         class updates.
      6. Numbers are session-scoped and NOT renumbered: an instance idle
         < evict_s that returns at a location with IoU < iou_min free-fills
         PAST the still-reserved rank (a newcomer takes #3 while #2 stays
         reserved) and keeps that number after the reservation is evicted,
         so two same-class instances can persistently display '#1'/'#3'
         with no '#2' (and with >2 instances the gaps can grow). This is
         deliberate -- renumbering would reintroduce the visible mid-stare
         '#1'/'#2' swap the reservations exist to stop. Tests and box-trial
         expectations must not read a persistent gap as a failure.
    """

    def __init__(self, iou_min=RANK_IOU_MIN, evict_s=RANK_EVICT_S,
                 dup_iou_min=DUP_IOU_MIN):
        self._iou_min = iou_min
        self._evict_s = evict_s
        self._dup_iou_min = dup_iou_min
        self._state = {}   # rank -> [t, xyxy]

    def _prune(self, class_name, now):
        state = self._state.get(class_name)
        if not state:
            return state
        for rank in [r for r, (t0, _b) in state.items()
                     if (now - t0) > self._evict_s]:
            del state[rank]
        return state

    def update(self, class_name, boxes, now):
        state = self._prune(class_name, now)
        if state is None:
            state = self._state[class_name] = {}
        pre = set(state)                      # pre-frame ranks
        ordered = sorted(enumerate(boxes or ()),
                         key=lambda eb: (-float(eb[1][2]), eb[0]))
        # dup fusion: drop boxes overlapping a
        # higher-conf one at IoU >= dup_iou_min
        kept = []
        for _idx, (key, xyxy, conf) in ordered:
            drop = False
            for _k2, keep_xyxy, _c2 in kept:
                if self._dup_iou_min is not None and \
                        box_iou(xyxy, keep_xyxy) >= self._dup_iou_min:
                    drop = True
                    break
            if not drop:
                kept.append((key, xyxy, conf))
        # carryover, conf desc; equal IoU →
        # lowest rank
        busy = set()                          # taken this frame
        out = {}                              # key -> (rank, carried)
        pending = []
        for key, xyxy, conf in kept:
            best_iou = None
            best_rank = None
            for rank in sorted(state):
                if rank in busy:
                    continue
                iou = box_iou(xyxy, state[rank][1])
                if iou >= self._iou_min and \
                        (best_iou is None or iou > best_iou or
                         (iou == best_iou and rank < best_rank)):
                    best_iou = iou
                    best_rank = rank
            if best_rank is not None:
                busy.add(best_rank)
                out[key] = (best_rank, True)
            else:
                pending.append((key, xyxy, conf))
        # free rank: smallest positive not taken
        # this frame, not reserved by a live rank
        for key, xyxy, _conf in pending:
            free = 1
            while free in busy or free in pre:
                free += 1
            busy.add(free)
            out[key] = (free, False)
        # slot refreshed
        for key, xyxy, _conf in kept:
            rank = out[key][0]
            state[rank] = [now, xyxy]
        return [(key,) + out[key] for key, _x, _c in boxes or ()
                if key in out]


# ── colour over sticky-mask px ────────────────────────────────────────────────
def _hue_ok(hue):
    try:
        hue = float(hue)
    except (TypeError, ValueError):
        return None
    if hue != hue:      # NaN caught here
        return None
    return hue % 360.0


def _band_contains(hue, lo, hi):
    """Half-open containment of a [0,360) hue in a band (wrap-safe)."""
    return ((hue - lo) % 360.0) < (hi - lo)


def _band_matches_tol(hue, lo, hi, tol):
    """Containment after expanding the band by tol on both edges (wrap-safe)."""
    return ((hue - (lo - tol)) % 360.0) < ((hi - lo) + 2.0 * tol)


def dominant_color(rows_hsv):
    """Dominant color stats over sticky-mask pixels. -> dict.

    rows_hsv: iterable of (h, s, v) rows, h in degrees [0, 360), s/v in
    [0, 1] fractions (see module docstring). Pure, no cv2.

    Returns dict with keys:
      hue             float|None  -- circular (vector) mean over the valid
                                     pixels, degrees, rounded to 3 dp;
                                     None whenever the color is untrusted
      sat_frac        float  -- fraction of ALL rows with s >= SAT_PIX_MIN
      color_frac      float  -- fraction of ALL rows with s >= SAT_PIX_MIN and
                                v >= V_PIX_MIN (the pixels the hue uses)
      color_name_hint str    -- band name containing the mean hue, else
                                'unknown' (display-only hint, never matched)
      n_px            int    -- number of usable rows
      reason          str    -- 'ok' | 'no_pixels' | 'low_px' | 'low_color_frac'
                                | 'low_purity' | 'no_band'

    Honest-unknown gates (in order): zero usable rows -> reason 'no_pixels'
    (distinct from too-few: no rows at all, e.g. a mask-less input);
    fewer than MIN_COLOR_PX usable rows -> 'low_px'; color_frac <
    COLOR_FRAC_MIN (low-sat / gray / dark object) -> 'low_color_frac';
    purity < HUE_PURE_FRAC_MIN (bimodal/multicolor object -- the vector
    mean would be garbage, so hue is reported None) -> 'low_purity'; hue
    falling in no band (can't happen while the table tiles, kept as a
    guard) -> 'no_band'. sat_frac/color_frac are per-pixel FRACTION
    statistics over the rows (never means/medians) and are always reported,
    unknown or not.
    """
    rows = []
    for row in rows_hsv or ():
        try:
            h, s, v = float(row[0]), float(row[1]), float(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if h != h or s != s or v != v:
            continue
        rows.append((h % 360.0, s, v))
    n = len(rows)
    if n:
        sat_frac = sum(1 for _h, s, _v in rows if s >= SAT_PIX_MIN) / float(n)
        chrom = [(h, s, v) for h, s, v in rows
                 if s >= SAT_PIX_MIN and v >= V_PIX_MIN]
        color_frac = len(chrom) / float(n)
    else:
        sat_frac = 0.0
        color_frac = 0.0
        chrom = []
    out = {'hue': None, 'sat_frac': sat_frac, 'color_frac': color_frac,
           'color_name_hint': 'unknown', 'n_px': n, 'reason': 'no_pixels'}
    if n == 0:
        # zero usable rows: 'no_pixels', not 'low_px'
        # (otherwise 0 < 30 would swallow it)
        return out
    if n < MIN_COLOR_PX:
        out['reason'] = 'low_px'
        return out
    if color_frac < COLOR_FRAC_MIN:
        out['reason'] = 'low_color_frac'
        return out
    sx = sum(math.cos(math.radians(h)) for h, _s, _v in chrom)
    sy = sum(math.sin(math.radians(h)) for h, _s, _v in chrom)
    mean_hue = (math.degrees(math.atan2(sy, sx)) % 360.0)
    within = 0
    for h, _s, _v in chrom:
        d = abs((h - mean_hue) % 360.0)
        if d > 180.0:
            d = 360.0 - d
        if d <= HUE_PURE_RADIUS_DEG:
            within += 1
    if (within / float(len(chrom))) < HUE_PURE_FRAC_MIN:
        out['reason'] = 'low_purity'
        return out
    name = 'unknown'
    for bname, lo, hi in HUE_BANDS:
        if _band_contains(mean_hue, lo, hi):
            name = bname
            break
    out['hue'] = round(mean_hue, 3)
    out['color_name_hint'] = name
    out['reason'] = 'ok' if name != 'unknown' else 'no_band'
    return out


def color_matches(hue, name, tol=COLOR_TOL_DEG):
    """Does a measured hue (degrees, None allowed) match the color NAME?

    Match = hue inside the name's band expanded by tol on both edges, with
    wraparound (so the red band hugs the 0/360 seam). hue None (no trusted
    color) -> False; 'unknown' never matches; unknown names never match.
    Bands tile, so at shared edges a hue can legitimately match both colours
    within tolerance (D4 continuous hue -- the caller picks by preference).
    """
    if name is None:
        return False
    name = str(name).strip().lower()
    if name == 'unknown':
        return False
    band = HUE_BAND_BY_NAME.get(name)
    if band is None:
        return False
    hue = _hue_ok(hue)
    if hue is None:
        return False
    lo, hi = band
    return _band_matches_tol(hue, lo, hi, tol)


def nearest_color_name(hue):
    """Nearest HUE_BANDS name to the hue (circular); 'unknown' for None/NaN."""
    hue = _hue_ok(hue)
    if hue is None:
        return 'unknown'
    best_name = None
    best_d = None
    for name, lo, hi in HUE_BANDS:
        centre = ((lo + hi) / 2.0) % 360.0
        d = abs(hue - centre) % 360.0
        if d > 180.0:
            d = 360.0 - d
        if best_d is None or d < best_d:
            best_d = d
            best_name = name
    return best_name


def frame_side(cx, img_w=IMG_W):
    """Image-frame side of a box centre: cx >= w/2 -> 'right', else 'left'."""
    try:
        return 'right' if float(cx) >= float(img_w) / 2.0 else 'left'
    except (TypeError, ValueError):
        return 'left'


# ── resolving the pick-target grammar ─────────────────────────────────────────
def _det_t(d):
    """Freshness stamp of a det dict.

    Two keys are honoured: 't' (the coordinator's cross-compat key, stamped
    alongside '_t' in pick_walk_coordinator._on_detections) and '_t' (pick_node's
    own receive-stamp key, stamped without 't' in _on_detections). Both are
    time.time() floats at receive; None (or float-ing garbage) -> not fresh.
    """
    for key in ('t', '_t'):
        try:
            return float(d.get(key))
        except (TypeError, ValueError):
            continue
    return None


def _det_conf(d):
    try:
        return float(d.get('conf'))
    except (TypeError, ValueError):
        return None


def _cand_side(d):
    s = d.get('body_side')
    if s is None:
        s = d.get('side')      # fallback key when body_side is absent
    return str(s or '').lower() if s is not None else None


def _render_visible(d):
    """'bottle#1(x=+0.04, right, color=orange)' style instance description."""
    label = str(d.get('label') or '?')
    comps = []
    try:
        x = float(d.get('x'))
        comps.append('x=%+.2f' % x)
    except (TypeError, ValueError):
        pass
    side = _cand_side(d)
    if side is None:
        side = d.get('frame_side')
    if side:
        comps.append(str(side).lower())
    comps.append('color=%s' % str(d.get('color_name_hint') or 'unknown'))
    return '%s(%s)' % (label, ', '.join(comps))


