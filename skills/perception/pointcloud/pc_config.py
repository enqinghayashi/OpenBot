"""PcConfig: every threshold the point-cloud consumer uses, in one place.

Python 3.8-safe (the box runs ROS 2 Foxy; py3.8 is INFERRED).  stdlib only.

Each field carries a label in ``LABELS``:
  MEASURED  - read from a file / log, or computed by running code
  DERIVED   - arithmetic on MEASURED values
  INFERRED  - rests on a stated assumption; REPLACE from a box capture
  POLICY    - a design choice (a gate, a bound), not a physical quantity

Calibration (docs/POINTCLOUD_CONSUMER_DESIGN.md section 10) replaces INFERRED
defaults with MEASURED values from capture_clouds.py output.  A calibrated value
must carry its capture filename in LABELS.
"""
import dataclasses
import hashlib


@dataclasses.dataclass(frozen=True)
class PcConfig:
    # ---------------------------------------------------------------- pairing rows
    PAIR_MAX_GAP_S: float = 0.15
    PAIR_BOX_PAD_M: float = 0.02
    ROW_MAX_AGE_S: float = 1.5
    ASSOC_GATE_M: float = 0.10
    MAX_CLOUD_POINTS: int = 20000

    # ---------------------------------------------------------------- pose tolerances
    POSE_Z_TOL_M: float = 0.01
    POSE_ANG_TOL_RAD: float = 0.0175
    POSE_MAX_AGE_S: float = 1.0
    POSE_REQUIRE_MEASURED: bool = True

    # ---------------------------------------------------------------- P1 hold detection
    HOLD_BASE_FRAMES: int = 5
    HOLD_BASE_MIN_FRAMES: int = 3
    HOLD_MIN_N0: int = 30
    HOLD_MIN_H0_M: float = 0.04
    HOLD_MIN_MEDIAN_RISE: float = 0.2
    HOLD_R_MIN_M: float = 0.02
    HOLD_FOOT_PAD_M: float = 0.04
    HOLD_FOOT_ZLO_M: float = 0.03
    HOLD_FOOT_ZHI_M: float = 0.05
    HOLD_CHK_FRAMES: int = 5
    HOLD_CHK_MIN_FRAMES: int = 3
    HOLD_CHK_DELAY_S: float = 2.5
    HOLD_MIN_ABS: int = 30
    HOLD_MIN_FRAC: float = 0.25
    HOLD_HIT_FRAMES: int = 3
    HOLD_BASE_DZ_M: float = 0.02
    HOLD_TOP_DZ_M: float = 0.03
    HOLD_SPAN_FRAC: float = 0.6
    HOLD_SHAPE_TOL: float = 0.35
    HOLD_STILL_XY_M: float = 0.03
    HOLD_MOVED_XY_M: float = 0.05
    HOLD_MOVED_MAX_XY_M: float = 0.20
    HOLD_MOVED_DZ_M: float = 0.03
    HOLD_MOVED_CONSIST_M: float = 0.03
    HOLD_OTHER_EXCL_M: float = 0.03
    HOLD_ABSENT_MIN_FRAMES: int = 4
    HOLD_MAX_SPAN_S: float = 60.0
    HOLD_OCCL_MIN_PTS: int = 20
    HOLD_PARTIAL_MIN_PTS: int = 10

    # ---------------------------------------------------------------- P2 bowl fitting
    BOWL_R_MIN_M: float = 0.05
    BOWL_R_MAX_M: float = 0.18
    BOWL_R_WIDTH_TOL: float = 0.30
    BOWL_CENTER_GATE_M: float = 0.04
    BOWL_SWEEP_STEP_M: float = 0.005
    BOWL_SLICE_HALF_M: float = 0.006
    BOWL_RANSAC_ITERS: int = 60
    BOWL_RIM_TOL_M: float = 0.008
    BOWL_RIM_INLIER_FRAC: float = 0.5
    BOWL_RIM_MIN_ARC_DEG: float = 120.0
    BOWL_RIM_MIN_PTS: int = 25
    BOWL_RIM_ABOVE_MEDIAN_MAX_M: float = 0.12
    BOWL_RIM_BELOW_MEDIAN_MAX_M: float = 0.03
    BOWL_NOISE_UNKNOWN_M: float = 0.005
    BOWL_MARGIN_K: float = 1.0
    BOWL_FLOOR_R_FRAC: float = 0.5
    BOWL_FLOOR_MIN_DEPTH_M: float = 0.02
    BOWL_FLOOR_MIN_PTS: int = 30
    BOWL_MIN_PTS: int = 80
    BOWL_EDGE_PX: float = 2.0
    BOWL_EDGE_MIN_PTS: int = 10
    BOWL_FRAMES: int = 3
    BOWL_TOP_GAP_M: float = 0.015
    BOWL_TOP_ABOVE_MAX_PTS: int = 15
    PLACE_CLEARANCE_M: float = 0.02
    COMFORT_Z: float = 0.0
    DEEP_FLOOR_Z: float = -0.54

    # ---------------------------------------------------------------- P3 split detection
    SUPPORT_LABELS: tuple = ("book", "laptop", "keyboard")
    SPLIT_OVERLAP_FRAC: float = 0.3
    SUPPORT_MAX_H_M: float = 0.10
    SPLIT_SLAB_M: float = 0.01
    SPLIT_WIDTH_RATIO: float = 1.6
    SPLIT_LEDGE_MAX_SLABS: int = 2
    SPLIT_MIN_UPPER_H_M: float = 0.05
    SPLIT_UPPER_TO_LOWER_H: float = 1.5
    SPLIT_MIN_SLAB_PTS: int = 5
    SPLIT_MARGIN_M: float = 0.01
    SPLIT_MIN_ABOVE_PTS: int = 30
    SPLIT_MIN_PTS: int = 60

    # ---------------------------------------------------------------- P4 grasp geometry
    GRASP_Z_OFFSET: float = 0.04
    GRASP_BAND_M: float = 0.02
    GRASP_MIN_BAND_PTS: int = 40
    GRASP_X_OFF_K: float = 0.065
    HAND_CORRIDOR_W_M: float = 0.10
    CORRIDOR_BLOCK_PTS: int = 20
    WIDTH_Q_LO: float = 5.0
    WIDTH_Q_HI: float = 95.0

    # ---------------------------------------------------------------- P5 person checks
    PERSON_MIN_PTS: int = 50
    PERSON_MIN_HEIGHT_M: float = 0.30
    PERSON_W_MIN_M: float = 0.20
    PERSON_MID_AGREE_M: float = 0.04
    PERSON_W_MAX_M: float = 1.2
    PERSON_EDGE_PX: float = 3.0
    PERSON_EDGE_MIN_PTS: int = 10
    PERSON_MIN_FRAMES: int = 2
    PERSON_POOL_N: int = 5
    PERSON_SLAB_M: float = 0.05

    # ---------------------------------------------------------------- node resource caps
    NODE_TTL_MAX_S: float = 15.0
    NODE_TTL_MIN_S: float = 0.5
    NODE_WARM_TTL_MAX_S: float = 20.0
    NODE_MIN_PERIOD_S: float = 0.2
    NODE_CB_BUDGET_S: float = 0.030
    NODE_RSS_CAP_MB: float = 400.0
    HOLD_BASE_KEEP_S: float = 120.0
    HOLD_BASE_MAX_KEPT: int = 4


LABELS = {
    "PAIR_MAX_GAP_S": "POLICY: body is published before the cloud in ONE callback "
                      "(projector :459 then :484); 0.15 s ~ one callback gap p50 0.122-0.128 s (MEASURED run20)",
    "PAIR_BOX_PAD_M": "POLICY (tight per critique; row xyz = R@median(pts_cam)+T, pre-stride)",
    "ROW_MAX_AGE_S": "POLICY (row 'stamp' is yolo inference wall time, yolo :407)",
    "ASSOC_GATE_M": "POLICY",
    "MAX_CLOUD_POINTS": "MEASURED projector :37 MAX_CLOUD_POINTS = 20000",
    "POSE_Z_TOL_M": "POLICY (1 cm, critique geometry-and-noise)",
    "POSE_ANG_TOL_RAD": "POLICY (1 deg)",
    "POSE_MAX_AGE_S": "POLICY",
    "POSE_REQUIRE_MEASURED": "POLICY: a COMMANDED height cannot see the writer's 66 s re-stand (HRC-6/7)",
    "HOLD_CHK_DELAY_S": "DERIVED: yolo candidate lifetime 2.0 s (yolo :382 '> 2.0') + 0.5 s margin (POLICY)",
    "HOLD_MAX_SPAN_S": "DERIVED: writer height_hold_s 66.0 s (MEASURED box) minus 6 s margin (POLICY)",
    "HOLD_BASE_DZ_M": "INFERRED floor; the live tolerance is max(this, 3*sigma of baseline per-frame p5z)",
    "HOLD_SPAN_FRAC": "POLICY: a bottle has vertical extent; a ghost-mask table patch is flat",
    "BOWL_CENTER_GATE_M": "POLICY; widened 0.03 -> 0.04 after the synthetic row-median bias was MEASURED (see design 4.2)",
    "BOWL_EDGE_MIN_PTS": "POLICY: a bowl touching the image border is refused (synthetic clipped bowls fit rims up to 0.022 m LOW: MEASURED by test_pc_core A-P2-7 exploration)",
    "HOLD_MIN_MEDIAN_RISE": "POLICY: baseline median z must rise >= 0.2*h0 above its p5 (a table-leak baseline around a transparent bottle does not)",
    "BOWL_NOISE_UNKNOWN_M": "POLICY, synthetic-calibrated (test_pc_core noise sweep): the ring sigma estimate reads <=0.004 at true depth sigma <=5 mm and >=0.0055 at >=15 mm, where the rim bias exceeds +0.01; above it the fit abstains",
    "BOWL_TOP_GAP_M": "POLICY: a fitted level more than 1.5 cm below on-or-outside-circle points is not the rim (review r1 HIGH)",
    "BOWL_TOP_ABOVE_MAX_PTS": "POLICY, synthetic-calibrated: n_above was 0 on all 455 within-1-cm fits and >=1373 on all 11 wall-circle fits (2 noise x 4 depth x 3 y x 2 x x 3 rim x 6 seeds)",
    "BOWL_MARGIN_K":"POLICY: drop_z adds max(0, K*sigma_rim) so a low-biased rim cannot drive the hand in",
    "PLACE_CLEARANCE_M": "MEASURED gateway/place_target.py:74 PLACE_CLEARANCE_M = 0.02",
    "COMFORT_Z": "MEASURED brain/descent_law.py:99 COMFORT_Z = 0.0 (copy, not import)",
    "DEEP_FLOOR_Z": "MEASURED brief: z<-0.54 DEEP_FLOOR (copy)",
    "SUPPORT_LABELS": "MEASURED yolo :137 TRACKED_CLASSES contains book/laptop/keyboard",
    "SUPPORT_MAX_H_M": "POLICY: applied to the SUPPORT SEGMENT's own z-span, not row height_m "
                       "(book rows read 0.46/1.27, lens 3 MEASURED)",
    "GRASP_Z_OFFSET": "MEASURED pick_engine_v3.py GRASP_Z_OFFSET = 0.04 (copy)",
    "GRASP_X_OFF_K": "MEASURED pick_engine_v3.py '# x_off trailing token - compute_x_off = -(r + 0.065)' (copy)",
    "WIDTH_Q_LO": "DERIVED: p5-p95 of a uniform-in-x silhouette spans 0.90 of 2r; estimator divides by 0.90",
    "PERSON_W_MIN_M": "INFERRED: two legs (sep 0.20 + 2*0.065) read >= ~0.25; ONE visible leg reads ~0.13 and is refused",
    "PERSON_MID_AGREE_M": "POLICY: extent-midpoint and slab-median must agree, else asymmetric (one leg hidden / side bag)",
    "PERSON_MIN_HEIGHT_M": "MEASURED gateway_node.py TURN_PERSON_MIN_HEIGHT_M = 0.30 (copy)",
    "PERSON_EDGE_MIN_PTS": "POLICY: a silhouette TOUCHING the left/right image border is clipped (the design's 20 %-of-points rule could never fire: a clipped edge is ~3 px of a ~100 px person)",
    "NODE_CB_BUDGET_S": "POLICY (critique integration-and-cpu: skip frames over 30 ms)",
    "NODE_RSS_CAP_MB": "POLICY (box OOM history; MEASURED FUTURE_WORK file :262-264)",
    "HOLD_BASE_KEEP_S": "POLICY (a pick plus a lane hold)",
}
DEFAULT_LABEL = "INFERRED (synthetic default; replace from capture)"


def label_of(name):
    return LABELS.get(name, DEFAULT_LABEL)


def cfg_hash(cfg):
    items = sorted(dataclasses.asdict(cfg).items())
    return hashlib.md5(repr(items).encode("utf-8")).hexdigest()
