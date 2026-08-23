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


def _fmt(xyxy):
    return '[%s]' % ','.join('%.0f' % v for v in xyxy)


def _centre(xyxy):
    return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)


def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


class _Track(object):
    __slots__ = ('iid', 'xyxy', 'first_seen', 'last_seen', 'label',
                 'challenger', 'streak')

    def __init__(self, iid, xyxy, now, label):
        self.iid = iid
        self.xyxy = xyxy
        self.first_seen = now
        self.last_seen = now
        self.label = label          # the class we publish now
        self.challenger = None      # different class leading right now
        self.streak = 0             # observations in a row


class InstanceTracker(object):
    def __init__(self, iou_min=0.5, dup_iou=0.7, evict_s=2.0,
                 centroid_gate_px=60, sticky_k_base=10, sticky_k_extra=3,
                 base=None, id_source=None, min_new_conf=0.35,
                 contain_frac=0.7, extra_floor=0.35):
        self.iou_min = float(iou_min)
        self.dup_iou = float(dup_iou)
        self.evict_s = float(evict_s)
        self.centroid_gate_px = float(centroid_gate_px)
        self.sticky_k_base = max(1, int(sticky_k_base))
        self.sticky_k_extra = max(1, int(sticky_k_extra))
        self.base = frozenset(base or ())
        self._ids = _PROCESS_IDS if id_source is None else id_source
        self._tracks = {}           # iid -> _Track; insertion order = age
        self.min_new_conf = float(min_new_conf)
        self.contain_frac = float(contain_frac)
        # admit floor for an operator extra (bin / box)
        # at or above it, an extra beats an overlapping BASE box
        # in dup fusion; at equal score the extra beats the base box
        # (bin 0.90 vs handbag 0.21 on one box)
        # node keeps this = its own admit floor
        self.extra_floor = float(extra_floor)
        # optional callable(str): every fusion removal + why
        # (OPENBOT_YOLOE_SCORE_TAP).  None = silent
        self.tap = None

    # ── query methods ────────────────────────────────────────────────────
    def __len__(self):
        return len(self._tracks)

    def ids(self):
        return list(self._tracks)

    def age(self, iid):
        """last_seen - first_seen of a live track (seconds), else None."""
        t = self._tracks.get(iid)
        return None if t is None else t.last_seen - t.first_seen

    def label(self, iid):
        """The track's published (sticky) class, else None."""
        t = self._tracks.get(iid)
        return None if t is None else self._label(t)

    majority = label        # old name; callers still use it

    @staticmethod
    def _label(t):
        return t.label

    def _observe(self, t, cls):
        """Fold one observed class into the track's sticky label."""
        if cls == t.label:
            t.challenger = None
            t.streak = 0
            return
        if cls == t.challenger:
            t.streak += 1
        else:
            t.challenger = cls
            t.streak = 1
        k = (self.sticky_k_base if cls in self.base
             else self.sticky_k_extra)
        if t.streak >= k:
            t.label = cls
            t.challenger = None
            t.streak = 0

    # ── fusing boxes ─────────────────────────────────────────────────────
    def _box_area(self, xyxy):
        """Area of a box (x1, y1, x2, y2)."""
        return (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

    def _intersection_area(self, a, b):
        """Intersection area of two boxes."""
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy

    def _fuse_key(self, boxes, i):
        """Dup-fusion priority of box ``i``: an active extra at or above its
        admit floor first, then the base classes, then sub-floor extras;
        within a rank higher conf, then input order.  Base against base and
        extra against extra order exactly as before."""
        cls, conf = boxes[i][0], float(boxes[i][2])
        if cls in self.base:
            rank = 1
        else:
            rank = 0 if conf >= self.extra_floor else 2
        return (rank, -conf, i)

    def fuse(self, boxes):
        """Indices of ``boxes`` kept after dup fusion (IoU >= dup_iou) and
        SAME-CLASS containment fusion (intersection/own_area >= contain_frac
        inside a higher-conf box of the same class: the F4 ghost boxes,
        MEASURED on Stuart 2026-10-06)."""
        # pass 1: containment fusion into higher-conf boxes
        order = sorted(range(len(boxes)),
                       key=lambda i: self._fuse_key(boxes, i))

        # containment check vs higher-conf boxes
        to_remove = set()
        for i in range(len(boxes)):
            if i in to_remove:
                continue
            # any higher-conf box containing i
            for j in range(len(boxes)):
                if i == j or j in to_remove:
                    continue
                if float(boxes[j][2]) <= float(boxes[i][2]):
                    # not more confident, skip
                    continue
                if boxes[j][0] != boxes[i][0]:
                    # SAME class only: a ghost / partial box of one object
                    # a different-class box inside a larger one is a real
                    # object ON or IN it (wallet on a book, cup in a bowl)
                    continue
                # contained? intersection / i.area >= contain_frac
                inter = self._intersection_area(boxes[i][1], boxes[j][1])
                own_area = self._box_area(boxes[i][1])
                if own_area > 0.0:
                    frac = inter / own_area
                    if frac >= self.contain_frac:
                        to_remove.add(i)
                        if self.tap is not None:
                            self.tap('[score_tap] fuse DROP box=%d %s '
                                     'conf=%.3f xyxy=%s: contained %.2f in '
                                     'same-class box=%d conf=%.3f'
                                     % (i, boxes[i][0], float(boxes[i][2]),
                                        _fmt(boxes[i][1]), frac, j,
                                        float(boxes[j][2])))
                        break

        kept = [i for i in range(len(boxes)) if i not in to_remove]

        # pass 2: dup fusion over the survivors
        order = sorted(kept, key=lambda i: self._fuse_key(boxes, i))
        final_kept = []
        for i in order:
            xyxy = boxes[i][1]
            hit = None
            for k in final_kept:
                v = iou_xyxy(boxes[k][1], xyxy)
                if v >= self.dup_iou:
                    hit = (k, v)
                    break
            if hit is not None:
                if self.tap is not None:
                    k, v = hit
                    self.tap('[score_tap] fuse DROP box=%d %s conf=%.3f '
                             'xyxy=%s: IoU %.2f with kept box=%d %s '
                             'conf=%.3f (rank %d vs %d)'
                             % (i, boxes[i][0], float(boxes[i][2]),
                                _fmt(xyxy), v, k, boxes[k][0],
                                float(boxes[k][2]),
                                self._fuse_key(boxes, i)[0],
                                self._fuse_key(boxes, k)[0]))
                continue
            final_kept.append(i)
        return sorted(final_kept)

    # ── the main entry points ────────────────────────────────────────────
    def update(self, boxes, now):
        """[(class_name, xyxy, conf)] -> [(iid, class_name, xyxy, conf,
        carried)] in input order (fused-away boxes omitted)."""
        return [r[1:] for r in self.update_indexed(boxes, now)]

