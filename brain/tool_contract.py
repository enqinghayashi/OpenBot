"""Tool contracts: what each skill tool may/must do, how to verify it, and a precondition gate for the agent (stdlib only)."""
from __future__ import annotations  # py3.8: list[str]/dict[...] evaluate at import
                                    # -> 'type' object is not subscriptable
                                    # kills the brain driver AND gateway _verify
                                    # (lazy import, inside try/except) ->
                                    # every motion verdict decodes as VERDICT:error

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContract:
    """Machine-readable contract for one skill tool (pick/place/navigate/open_hand/close_hand).

    Fields (exact order per shared contract):
      name            - CONTRACTS key, e.g. 'pick'
      preconditions   - statements that must hold BEFORE invocation; those with
                        keywords in _PRECONDITION_KEYWORDS are judgeable from a
                        scene_brief, the rest are runtime checks the agent does directly
      effects         - world-state changes the tool is expected to cause
      post_condition  - single truth statement the verifier must establish
      verify_query    - what the verifier queries/checks (in platform vocabulary)
      timeout_s       - upper bound the agent uses for watchdog/timeout planning
      context         - free-form platform facts an agent needs to drive the tool
    """

    name: str
    preconditions: list[str]
    effects: list[str]
    post_condition: str
    verify_query: str
    timeout_s: float
    context: str


# ---------------------------------------------------------------------------
# scene-brief keywords per precondition
# guard: every CONTRACTS entry mapped
# () = runtime check, not judgeable
# != () = any keyword, unless negated
# 'unmet' = "brief didn't confirm it"
# ---------------------------------------------------------------------------

_PRECONDITION_KEYWORDS: dict[str, tuple[str, ...]] = {
    # pick
    "object on a reachable tabletop surface": ("tabletop", "reachable"),
    "object pose known from a fresh detection": ("fresh", "detected", "detection"),
    "gripper not already holding the target object": (),
    "no concurrent locomotion command (writer precedence law: hold_rc/shutdown > cmd_auto > live RC)": (),
    # place
    "object held in gripper (grip contact == True)": ("holding", "contact"),
    "robot at the intended drop location (walk_status REACHED if an approach was needed)": ("drop", "reached"),
    # navigate
    "walk FSM reachable (walk_status topic live)": ("walk", "fsm"),
    "target location known relative to torso frame": ("target", "goal", "waypoint"),
    # open_hand / close_hand
    "grip socket available and responding": (),
    "object within gripper closing volume (if grasp intended)": ("object", "grasp"),
    # pose
    "arm not holding a load that the target posture would drop or crush": (),
}


CONTRACTS: dict[str, ToolContract] = {
    "pick": ToolContract(
        name="pick",
        preconditions=[
            "object on a reachable tabletop surface",
            "object pose known from a fresh detection",
            "gripper not already holding the target object",
            "no concurrent locomotion command (writer precedence law: hold_rc/shutdown > cmd_auto > live RC)",
        ],
        effects=[
            "gripper holds target object (grip contact)",
            "object absent from source location",
        ],
        post_condition="object gone from source AND grip contact",
        verify_query=(
            "re-observe detections: target ref absent at source location; "
            "grip state via 'state <side>' -> contact == True "
            "(grip does NOT auto-verify - 'state' is THE verification signal)"
        ),
        # 180 s = CONTRACT_TIMEOUTS["pick"] (brain/agent/tools.py)
        # async: DONE ~3 s in, daemon moves ~33 s more
        # pinned by tests/test_pick_timeout_ladder.py
        timeout_s=180.0,
        context=(
            "pick engine (tabletop_pick_place_engine, unix socket "
            "/tmp/pick_engine.sock, NON-owning): pick <x> <y> <z> [nx ny nz] [orientation] [height_m], "
            "positive x = right arm; pick_left for left arm. Engine takes hold_rc on itself. "
            "TABLETOP-ONLY: workspace torso-frame z >= ~-0.35 m (z=0 is the torso link); "
            "floor/ground objects are a capability-block - "
            "refuse honestly, never attempt a floor pick (no ground-pick skill exists). "
            "Grip close_adaptive -> DONE:contact|full|no_data (contact = the real grasp signal). "
            "Human confirm gate: required before ANY motion (pre-hook)."
        ),
    ),
    "place": ToolContract(
        name="place",
        preconditions=[
            "object held in gripper (grip contact == True)",
            "robot at the intended drop location (walk_status REACHED if an approach was needed)",
        ],
        effects=[
            "object released from gripper",
            "gripper open",
            "payload cleared (payload off)",
            "hold_rc restored to off after drop",
        ],
        post_condition="grip open AND object out of hand",
        verify_query=(
            "grip state via 'state <side>' (side = the holding arm) -> open == True "
            "AND object not in hand; the verifier checks ONLY that the grip opened "
            "and the object left the hand — it does NOT verify the object landed at "
            "an optional x/y/z drop coordinate; never claim in-bin or at-coordinate "
            "without demonstrated containment"
        ),
        timeout_s=60.0,
        context=(
            "place carries an OPTIONAL x/y/z drop point (a commanded target): the "
            "engine place drops to the coordinate when present and releases in "
            "place when absent.  side = the holding arm.  The verifier still "
            "verifies only 'grip open AND object out of hand' — it does NOT verify "
            "the object landed at the coordinate. "
            "Human confirm gate: required before ANY motion (pre-hook)."
        ),
    ),
    "navigate": ToolContract(
        name="navigate",
        preconditions=[
            "walk FSM reachable (walk_status topic live)",
            "target location known relative to torso frame",
        ],
        effects=["robot displaced to target location"],
        post_condition="robot at target (walk_status == REACHED)",
        verify_query=(
            "walk_status phase == REACHED ONLY; never odom distance "
            "(/unitree/slam_mapping/odom does NOT exist)"
        ),
        timeout_s=60.0,
        context=(
            "ROS2 std_msgs/String topics: /walk_target, /walk_go (only the approach trigger), "
            "/walk_status (FSM SEARCHING/APPROACHING/REACHED). Navigate verifies on "
            "/walk_status == REACHED ONLY - there is no odom. NEVER sport RPC 3102 (refused while "
            "the writer owns the robot): locomotion ONLY via cmd_set/engine or walk FSM topics. "
            "Detections frame: torso, x = lateral RIGHT, y = forward; omega sign verified per run."
        ),
    ),
    "open_hand": ToolContract(
        name="open_hand",
        preconditions=["grip socket available and responding"],
        effects=["gripper open (open_both for both sides)"],
        post_condition="grip open (joints near open limit)",
        verify_query=(
            "grip state via 'state <side>' -> open == True (joints near open limit); "
            "grip does NOT auto-verify"
        ),
        timeout_s=10.0,
        context=(
            "Grip socket /tmp/grip_server.sock: open|open_left|open_both -> one reply line. "
            "Verify via 'state [side]' (joints + hand-state). Grip does NOT auto-verify."
        ),
    ),
    "close_hand": ToolContract(
        name="close_hand",
        preconditions=[
            "grip socket available and responding",
            "object within gripper closing volume (if grasp intended)",
        ],
        effects=["gripper closed; grip contact if an object is present"],
        post_condition="grip closed (contact or full-stroke stop)",
        verify_query=(
            "grip state via 'state <side>' -> closed (contact == True or full-stroke stop); "
            "grip does NOT auto-verify"
        ),
        timeout_s=15.0,
        context=(
            "Grip socket /tmp/grip_server.sock: close|close_adaptive <side> "
            "[max_fraction steps dt stop_threshold min_steps] -> DONE:contact|full|no_data "
            "(contact = the real grasp signal). Verify via 'state [side]'. "
            "close_adaptive is preferred for grasps (stop on contact evidence)."
        ),
    ),
    "pose": ToolContract(
        name="pose",
        preconditions=[
            "arm not holding a load that the target posture would drop or crush",
        ],
        effects=[
            "both arms move to the named posture (ready | tuck | neutral | home)",
            "no grip change: a pose never opens or closes a hand",
        ],
        # HONESTY BOUNDARY: no posture sensor
        # so the post-condition is the weakest
        # true thing, not the wanted one
        # see verify_pose
        post_condition=(
            "the named posture was commanded and the pick engine reported the "
            "move complete (engine-reported, NOT sensor-measured — no posture "
            "sensor exists)"
        ),
        verify_query=(
            "engine reply 'DONE:<v1>[+<v2>]' whose verb list covers the "
            "requested pose; no independent arm-joint observation is available"
        ),
        timeout_s=30.0,
        context=(
            "Reaches the engine through the gateway's GENERIC 'engine' tool "
            "(no gateway verb needed). Engine verbs: ready | carry+carry_left "
            "(= tuck) | neutral | home. 'tuck' is TWO sequential moves because "
            "the engine's carry pose is per-arm; the sequence is load-bearing. "
            "NOTE: the engine's own module docstring omits 'neutral' — the "
            "command dispatch is the authority, not the docstring."
        ),
    ),
    "turn_around": ToolContract(
        name="turn_around",
        # reuses navigate's precondition string verbatim
        # -> no new keyword entry, guard still passes
        # same walk FSM navigate talks to
        preconditions=[
            "walk FSM reachable (walk_status topic live)",
        ],
        effects=[
            "the robot turns in place, sweeping the room for the named label",
            "the walk FSM stops the sweep the moment the label is in view "
            "(the DETECTED hold) and does NOT approach",
        ],
        # HONESTY BOUNDARY
        # read verify_turn_around first
        # returns AS SOON AS the sweep is COMMANDED:
        # publish, reply, no wait
        # no completion event to observe
        # so: weakest true statement, same as 'pose'
        post_condition=(
            "the sweep was COMMANDED and its label published on /walk_target "
            "(gateway-reported, NOT sensor-measured — whether anyone is in "
            "view is NOT verified by this call; the FSM reports that later "
            "on /walk_status as DETECTED)"
        ),
        verify_query=(
            "gateway reply DONE turn_around: ... -> the publish was accepted. "
            "There is NO epoch and NO phase annotation on this reply, so "
            "unlike navigate the verifier cannot prove the FSM reacted to "
            "THIS dispatch — see verify_turn_around"
        ),
        # walk FSM's sweep bound, same as
        # CONTRACT_TIMEOUTS["navigate"]
        # gateway replies at once; bounds the sweep only
        timeout_s=180.0,
        context=(
            "Turns in place and sweeps for a person/label, then HOLDS on "
            "sight — it does NOT walk to them (that is navigate).  The "
            "gateway publishes the bare label DIRECTLY on /walk_target, "
            "never /navigate_target, so the pick/walk coordinator never sees "
            "a navigation request and its auto-approach can never fire. "
            "The verb takes an OPTIONAL label (default 'person'). "
            "Human confirm gate: required before ANY motion (pre-hook)."
        ),
    ),
}

# every CONTRACTS entry needs a keyword map entry
_UNMAPPED = [
    pc for c in CONTRACTS.values() for pc in c.preconditions if pc not in _PRECONDITION_KEYWORDS
]
if _UNMAPPED:
    raise RuntimeError(f"tool_contract: preconditions missing keyword map entry: {_UNMAPPED}")

_NEGATION_RE = re.compile(r"\b(no|not|none|never)\s+(?P<kw>\w+)")


def validate_precondition(text: str) -> list[str]:
    """Return unmet preconditions of all contracts against a scene_brief `text`.

    Keyword heuristic: a scene-expressible precondition is 'met' when the brief contains
    any of its mapped keywords, unless negated ('no/not/none/never <keyword>'). Runtime
    preconditions (keyword map = ()) are never judged here - the agent checks them
    directly against the platform. 'Unmet' means "brief does not confirm it" -> the
    agent must re-observe, never guess. Empty text = empty scene -> nothing confirmed.

    Returns entries prefixed with the contract name, e.g. '[pick] object on a
    reachable tabletop surface'. Heuristic only - never a correctness claim.
    """
    if not text:
        return [
            f"[{c.name}] {pc}"
            for c in CONTRACTS.values()
            for pc in c.preconditions
            if _PRECONDITION_KEYWORDS[pc]
        ]
    lowered = text.lower()
    unmet: list[str] = []
    for contract in CONTRACTS.values():
        for pc in contract.preconditions:
            keywords = _PRECONDITION_KEYWORDS[pc]
            if not keywords:
                continue  # runtime precondition: the agent judges it
            satisfied = False
            for kw in keywords:
                pattern = rf"\b{re.escape(kw)}\w*\b"
                if re.search(pattern, lowered) and not _contradicted_by_negation(lowered, kw):
                    satisfied = True
                    break
            if not satisfied:
                unmet.append(f"[{contract.name}] {pc}")
    return unmet


