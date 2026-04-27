# brain/locate_tool.py - stdlib only, runs on py3.8
"""Optional brain-tool lane `spatial_lane` (B7, 2026-10-06): the `locate` tool.

ON only when ``optional_tools: {spatial_lane: true}`` (the batch-4 lane
convention, brain/agent/optional_tools.py); OFF = the tool list sent to the
model is byte-identical to today.

`locate` answers a DESCRIPTION ("the red cup next to the schoolbag", "the
leftmost bottle", "the cup behind the bowl") with ONE named object and its
measured torso-frame position, or with the list of candidates when the
description fits several, or honestly with none.  The geometry is
deterministic (brain/spatial.py) over the scene graph's FRESH, not-in-hand
objects; the model only fills the structured arguments.  An object without a
measured position is never placed at 0.

The returned ``name`` (``red_cup_2``) is what pick / place take as the label:
the scene graph resolves it to exactly that object (instance id, else its
stored position within the association gate) and refuses when that object is
not in view, never binding another same-class object.
"""
from typing import Any, List

from brain import spatial

LOCATE_DESCRIPTION = (
    "Find ONE object from a description with a colour, a side, a relation to "
    "another object, or a superlative -- e.g. 'the red cup next to the "
    "schoolbag', 'the green bottle on the left', 'the cup behind the bowl', "
    "'the leftmost book', 'the nearest cup'. Returns its name (use that name "
    "as the label for pick/place), its position x (right +) / y (forward) / "
    "z (up) in metres from the robot's torso, its distance and side; or the "
    "candidate names when several fit (then ask the user which one); or that "
    "nothing fits. Call it BEFORE pick whenever several objects of the same "
    "kind may be in view. Relations are from the robot's point of view: "
    "in_front_of = closer to the robot, behind = farther away."
)

LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string",
                  "description": "The object's kind, e.g. 'cup', 'bottle', 'backpack' "
                                 "(plain class word, no colour)."},
        "colour": {"type": "string",
                   "description": "Colour word, optional: red, orange, yellow, green, "
                                  "cyan, blue, purple, pink (teal, navy, maroon ... "
                                  "are understood)."},
        "side": {"type": "string", "enum": ["left", "right", "ahead"],
                 "description": "Which side of the robot, optional."},
        "relation": {"type": "string",
                     "enum": ["near", "next_to", "left_of", "right_of",
                              "in_front_of", "behind", "above", "below"],
                     "description": "Relation to the anchor object, optional."},
        "anchor_label": {"type": "string",
                         "description": "The anchor object's kind (with relation)."},
        "anchor_colour": {"type": "string",
                          "description": "The anchor object's colour, optional."},
        "superlative": {"type": "string",
                        "enum": ["leftmost", "rightmost", "nearest", "farthest",
                                 "highest", "lowest", "largest", "smallest"],
                        "description": "Pick the extreme one, optional."},
    },
    "required": ["label"],
}

LOCATE_TIMEOUT_S = 5.0


class Locate(object):
    """The tool body.  Reads registry.scene_graph / registry.backend at CALL
    time (lane factories run before the registry is attached)."""

    def __init__(self, registry):
        self.registry = registry

    def __call__(self, args):
        from brain.agent.tools import fail, ok
        reg = self.registry
        graph = getattr(reg, "scene_graph", None)
        if graph is None:
            return fail("scene graph not wired - cannot locate")
        if getattr(reg, "backend", None) is not None:
            obs = reg._observe()
            if obs is None or not getattr(obs, "fresh", True):
                return fail("NOT_DETECTED: no fresh observation - perception is "
                            "not answering; nothing can be located")
            graph.update(obs)
        objs = graph.spatial_objects()
        names = graph._compute_display_names()
        query = {"label": args.get("label") or "",
                 "colour": args.get("colour") or None,
                 "side": args.get("side") or None,
                 "superlative": args.get("superlative") or None}
        if args.get("relation") or args.get("anchor_label"):
            query["relation"] = {"type": args.get("relation") or "near",
                                 "anchor": {"label": args.get("anchor_label") or "",
                                            "colour": args.get("anchor_colour") or None}}
        res = spatial.resolve(objs, query)
        if res["status"] == "ok":
            o = res["obj"]
            deg, side = spatial.bearing(o)
            name = names.get(o["ref"], res.get("name") or o["ref"])
            z = ("z=%+.2f" % float(o["z"])) if o.get("z") is not None \
                else "height not measured"
            parts = ["%s: %s at x=%+.2f y=%.2f %s m, %.2f m away, %s"
                     % (name, o.get("label"), float(o["x"]), float(o["y"]), z,
                        spatial.distance(o), side)]
            if o.get("colour"):
                parts.append("colour %s" % o["colour"])
            parts.append("say pick %s to pick it" % name)
            return ok("; ".join(parts))
        if res["status"] == "ambiguous":
            return fail("ambiguous: %d objects fit (%s) - ask the user which one, "
                        "or add a colour, side or relation"
                        % (len(res["candidates"]), ", ".join(res["candidates"])))
        # bare NOT_DETECTED read as final, so it only apologised
        # ("I can't see a bottle"). not in view != unavailable
        label = str(args.get("label") or "").strip() or "it"
        return fail("NOT_DETECTED: %s. It is not in view right now - this is NOT "
                    "the end: call navigate('%s') to search the room and walk to "
                    "it, then pick it." % (res.get("reason", "nothing fits"), label))


def make_tools(registry: Any) -> List[Any]:
    """[locate] for register_optional_tools (lane `spatial_lane`)."""
    from brain.agent.tools import Tool
    handler = Locate(registry)
    tool = Tool(name="locate", description=LOCATE_DESCRIPTION, schema=LOCATE_SCHEMA,
                run=handler, timeout_s=LOCATE_TIMEOUT_S)
    tool.handler = handler  # hook for tests
    return [tool]
