"""
brain/embodiment.py — robot capability payload from the config.yaml
`embodiment:` segment, with STUART-platform defaults.

Stdlib only (brain/ cores carry no third-party deps — in particular NO
PyYAML).  Parsing is delegated ENTIRELY to brain/config_loader.py — the
ONE shared stdlib parser — including its value-level trailing-comment
stripping ('key: value  # comment' parses as `value`).  This module never
parses YAML itself: `from_config_file()` reads through
`config_loader.load()` and hands the parsed `embodiment:` dict to
`from_dict()`.  Full YAML is out of scope; if the segment ever needs
richer syntax, extend config_loader (or parse upstream and hand the
result to `from_dict()` instead).

Segment schema (all keys optional; missing segment or missing keys fall
back to these defaults):

    embodiment:
      envelope:                 # operational envelope
        pick_min_height_m: -0.35    # WORKSPACE FLOOR in the TORSO FRAME:
                                    # z is height ABOVE THE
                                    # TORSO LINK — the depth projector
                                    # (camera 0.46268 m above torso, 48 deg
                                    # tilt) and the pick engine (ik_z = torso_z)
                                    # both speak it, and live tabletop
                                    # detections read z in [0.00, +0.07].  A
                                    # ground object reads ~-0.75 (G1 torso is
                                    # ~0.75-0.8 m above the floor); -0.35
                                    # refuses every ground pick while any
                                    # tabletop/low-workspace object passes.
                                    # TABLETOP-ONLY. Ground objects are a
                                    # capability-block — the brain must refuse.
        pick_max_height_m: 1.20     # conservative upper bound (default; verify)
        max_reach_m: 0.90           # conservative arm reach (default; verify)
        grip_center_standoff_m: 0.0    # engines consume NO standoff (stdin
                                       # x/y/z = wrist target); the grip-zone
                                       # offset is handled engine-side.
        min_arm_dur_s: 1.5          # writer clamps arm <dur> to >= 1.5 s
      perception:
        freshness_s: 2.0            # agent.detection_freshness_s parity
        silence_means_empty: true   # /object_detections_body is event-driven:
                                    # silence == empty scene, NOT no-data
        frame: torso                # x = lateral RIGHT, y = forward, z = up
      base_relative:
        key_positions:              # named positions relative to base/torso
          grip_center_m: [0.0, 0.37, 0.0]   # torso-frame offset of the grip center
          camera_offset_m: [0.0, 0.0, 0.0]  # detections already torso-frame

Defaults encode verified platform facts where they exist; everything else
is marked "default; verify" — never presented as measured.

RUNTIME CONSUMERS (round-5 config-steering: ONLY verified read sites are
documented — a documentation map is NOT a consumer, so every entry below
is a real runtime read, wired by ``G1Agent.from_config`` into the RUNNING
hooks/tools):
  - envelope.pick_min_height_m / envelope.pick_max_height_m ->
    brain/hooks.py safety_checks_hook (floor/overhead refusals, pre-motion;
    envelope injected by brain/agent/agent.py G1Agent.from_config ->
    G1Agent.__init__ default_hooks envelope=) + brain/agent/tools.py
    ToolRegistry._constrained_pick_point (independent second net; registry
    embodiment wired by from_config).
  - envelope.max_reach_m -> brain/agent/tools.py
    ToolRegistry._constrained_pick_point (horizontal reach clamp).
  - envelope.grip_center_standoff_m -> brain/agent/tools.py
    ToolRegistry._constrained_pick_point (z/x grip-center standoff).
  - envelope.min_arm_dur_s -> gateway/gateway_node.py
    resolve_embodiment_config -> GatewayNode._min_arm_dur_s ->
    _clamp_arm_dur (arm-dispatch dur lower bound: the writer clamps
    `arm <14 floats> [dur]` to >= 1.5 s, so the EFFECTIVE floor is
    max(min_arm_dur_s, 1.5)).
  - perception.freshness_s -> brain/agent/agent.py G1Agent.from_config
    (fallback authority for detection_freshness_s when the agent segment
    is silent about it) -> brain/hooks.py safety_checks_hook (the
    pre-motion freshness budget).  Config.yaml parity convention: the
    agent segment's detection_freshness_s wins when present, the
    embodiment value is the runtime source otherwise.
  - perception.silence_means_empty -> gateway/gateway_node.py
    resolve_embodiment_config -> GatewayNode._silence_means_empty ->
    gateway/assemble.py Assembler (silence == empty scene, NOT no-data).
    SAFETY-ADJACENT: brain/config_loader.py rejects any non-bool value
    for this key, and config.yaml documents that it MUST stay True for
    the shipped profile.
  - perception.frame -> gateway/gateway_node.py resolve_embodiment_config
    -> GatewayNode._detections_frame (detections frame name; documents
    the cmd_set omega sign mapping — x = lateral RIGHT, y = forward,
    z = up).
  - base_relative.key_positions -> brain/agent/tools.py
    ToolRegistry.place_at_spot (named-spot lookup: the spot resolves to
    its CONFIGURED base-relative position; an unknown spot name is an
    honest tool error, never a guessed position).
Consumers read the fields off the injected object (plain attribute access
— ``emb.envelope.max_reach_m``) — the injected object is the loader's
validated ``EmbodimentConfig`` (brain/config_loader.py), which mirrors
this module's schema exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config_loader import load as _load_config

# lazy import
# try/except, gateway's old guard
# (gateway/gateway_node.py:1132-1135)
# no descent law?
# lane FORCED OFF
# dispatch keeps today's honest refusal
# never an import crash
# rollback: behaviour, not boot
try:
    from .descent_law import effective_pick_floor
except Exception as _exc:  # noqa: BLE001 — degrade, never raise
    print("[descent] descent_law import FAILED (%r) — low-workspace lane "
          "FORCED OFF" % (_exc,), flush=True)

    def effective_pick_floor(envelope):
        """Lane-OFF floor: the envelope's own ``pick_min_height_m`` UNCHANGED.

        The fallback used only when brain/descent_law.py is absent.  It
        reproduces the law's switch-off return exactly, so a partial deploy
        keeps today's honest capability-block refusal instead of widening or
        crashing.
        """
        return envelope.pick_min_height_m


@dataclass
class Envelope:
    """Operational envelope the brain must never exceed.

    Every field has a verified runtime consumer — see the RUNTIME
    CONSUMERS section of the module docstring (envelope.* rows)."""

    pick_min_height_m: float = -0.35  # torso-frame floor, 2026-09-06:
                                      # tabletop z in [0.00,+0.07],
                                      # ground ~ -0.75
    pick_max_height_m: float = 1.20  # conservative
                                     # verify on robot
    max_reach_m: float = 0.90  # conservative
                               # verify on robot
    grip_center_standoff_m: float = 0.0  # 2026-09-06
                                         # engines take no standoff
                                         # stdin x/y/z = wrist target,
                                         # grip offset is engine-side
    min_arm_dur_s: float = 1.5  # writer's arm <dur> clamp floor,
                                # verified; consumer: arm dispatch

    # ── low-workspace descent lane (plan §8) — copies EnvelopeConfig's fields
    # `low_workspace_lane` is the rollback switch:
    # False (default) answers as today
    # four law inputs
    # threaded from descent_law.py
    # by the gateway; no literal restated
    low_workspace_lane: bool = False  # switch to the descend lane;
                                      # strict bool only
    height_stand_m: float = 0.74  # nominal stand
                                  # (writer height_stand)
                                  # SIMULATED not measured
    height_min_m: float = 0.55    # writer's height clamp floor;
                                  # budget = stand - min
                                  # writer ships band shut
    hold_tol_m: float = 0.05      # hold tolerance, m
    descend_confirm_timeout_s: float = 3.0  # confirm poll, s
    lane_operation_budget_s: float = 30.0  # lane budget, s
                                           # mirrors descent_lane.LANE_OPERATION_BUDGET_S
                                           # a literal, not an import --
                                           # brain never imports gateway
                                           # not in summary()
    pre_descent_step: bool = False  # optional stance step pre-descent;
                                    # mirrors the EnvelopeConfig field;
                                    # strict bool, off
                                    # not in summary()
    left_hand_in_service: bool = False  # mirrors EnvelopeConfig
                                        # left_hand_in_service;
                                        # False = refuse left-arm picks

    def summary(self) -> str:
        if self.low_workspace_lane is True:
            # lane armed: floor from the law's
            # writer band, every call
            # never a literal
            # descent is writer-reported
            # not measured: no height sensor
            floor = effective_pick_floor(self)
            lane = (
                f"TABLEtop + low-workspace descent lane (writer-reported, "
                f"NOT sensor-measured; effective floor {floor:.2f} m)"
            )
        else:
            # lane off: byte-identical output
            floor = self.pick_min_height_m
            lane = "TABLEtop-only; floor pick = capability-block"
        return (
            f"pick z in [{floor:.2f},{self.pick_max_height_m:.2f}] m "
            f"({lane}), "
            f"reach <= {self.max_reach_m:.2f} m, "
            f"grip standoff {self.grip_center_standoff_m:.2f} m, "
            f"arm dur >= {self.min_arm_dur_s:.1f} s"
        )


@dataclass
class Perception:
    """Perception config governing how observations are interpreted.

    silence_means_empty is SAFETY-ADJACENT: brain/config_loader.py rejects
    any non-bool value for it, and config.yaml documents it MUST stay True
    for the shipped profile.  Consumers: see the RUNTIME CONSUMERS section
    of the module docstring (perception.* rows)."""

    freshness_s: float = 2.0  # detection staleness, agent parity
    silence_means_empty: bool = True  # silence = empty
    frame: str = "torso"  # x right, y forward
                          # z up

    def summary(self) -> str:
        return (
            f"frame={self.frame} (x right, y forward, z up), "
            f"freshness {self.freshness_s:.1f} s, "
            f"silence_means_empty={self.silence_means_empty}"
        )


@dataclass
class BaseRelative:
    """Named positions relative to the robot base/torso frame."""

    key_positions: Dict[str, List[float]] = field(default_factory=lambda: {
        "grip_center_m": [0.0, 0.37, 0.0],  # standoff, verified
        "camera_offset_m": [0.0, 0.0, 0.0],  # detections already torso-frame
    })

    def position(self, name: str) -> Optional[List[float]]:
        return self.key_positions.get(name)

    def summary(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in self.key_positions.items())
        return f"key positions: {items}"


def _as_bool(value: Any, name: str) -> bool:
    """Coerce a parsed perception value to bool; raise on anything else.

    The loader already returns Python bools; this also accepts the common
    string spellings so a hand-built dict cannot silently flip a
    safety-relevant flag (bool('false') == True would invert
    silence_means_empty).  Garbage raises — never a guess.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    low = str(value).strip().lower()
    if low in ("true", "yes", "on", "1"):
        return True
    if low in ("false", "no", "off", "0"):
        return False
    raise ValueError(f"embodiment perception.{name} must be a bool, got {value!r}")


@dataclass
class Embodiment:
    """Capability payload assembled from the `embodiment:` config segment."""

    envelope: Envelope = field(default_factory=Envelope)
    perception: Perception = field(default_factory=Perception)
    base_relative: BaseRelative = field(default_factory=BaseRelative)

