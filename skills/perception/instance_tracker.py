"""instance_tracker.py -- stable per-object instance ids for /detections (G1).

Batch 4 / S5 (docs/YOLOE_BATCH4_ARCHITECTURE.md section 6).  Used by
yolo_node_SEG_version.py ONLY when OPENBOT_INSTANCE_IDS=1; with the switch
absent the node keeps today's 30 px centroid-grid key and never imports this
module.

stdlib ONLY (no numpy, no rclpy) and Python 3.8 syntax: it ships hand-scp'd
co-located with the node on the box (Ubuntu 20.04 / py3.8).

Semantics
  * ``InstanceTracker.update(boxes, now)`` with ``boxes = [(class_name,
    xyxy, conf)]`` returns ``[(instance_id, class_name, xyxy, conf,
    carried)]`` in INPUT order, one tuple per box that survived dup fusion.
    ``class_name`` is the track's STICKY label (B8h): the first class seen
    is the incumbent, and a different class replaces it only after it has
    been the observed class for K CONSECUTIVE observations of that track
    (K = ``sticky_k_base`` = 10 when the challenger is a base class, K =
    ``sticky_k_extra`` = 3 when it is not in ``base``, i.e. an operator
    extra such as bin / box).  Any observation of the incumbent, or of a
    different challenger, restarts the streak.  Replaces the 5-vote
    majority, which let a static object flip ~0.5 times a second between
    near-synonym classes (s1 2026-10-08: backpack / suitcase / handbag).
    The id survives a label change either way.
    ``carried`` is True when the id existed before this frame.
  * Dup fusion BEFORE association: two boxes at IoU >= dup_iou collapse to
    one -- same class -> higher conf; different classes -> an active extra
    at or above ``extra_floor`` (B8h: it used to lose to any base class),
    else the base class; ties (both base / both extra) -> higher conf, then
    input order.  An extra below the floor still loses to a base class.
  * Association is CLASS-AGNOSTIC: boxes in conf-descending order (ties ->
    input order) each take the free live track with the highest IoU >=
    iou_min (ties -> the older id).  A box left unmatched may then take a
    free track whose last centroid is < centroid_gate_px away, but ONLY when
    that track is the single live track inside the gate AND no other box of
    this frame is inside the gate around that track (ambiguity check: a
    neighbour is never handed an id during a short occlusion).
  * A track not seen for MORE than evict_s (today's stale rule, node :432)
    is evicted at the start of the next update; a returning object gets a
    NEW id.  Ids come from ONE process-wide counter starting at 1 and are
    never reused within a process lifetime.
  * ``drop_classes(allowed)`` removes only the tracks whose sticky label
    is not in ``allowed`` (vocabulary swap, section 3.4 selective
    invalidation); every other id survives the swap.
"""
import itertools

# one id source per process
# ids never reused, even across a tracker rebuild
_PROCESS_IDS = itertools.count(1)


def iou_xyxy(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
          - inter)
    return inter / ua if ua > 0.0 else 0.0
