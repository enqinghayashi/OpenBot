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


