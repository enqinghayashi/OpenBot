"""look_for — the open-vocabulary lane's brain tool (batch 4, S4).

Switch: ``optional_tools.open_vocab_lane: true`` in the brain config (box
hand-merge, docs/BOX_MERGE_config_voice_instructions_v3.md).  OFF by default:
this module is imported ONLY by ``brain.agent.optional_tools`` when that
switch is exactly ``True``; with it absent nothing here is ever loaded and the
tool specs sent to the model are byte-identical to the base registry.

What it does (docs/YOLOE_BATCH4_ARCHITECTURE.md §5):

* ``make_tools(registry)`` returns ``[look_for, scene_brief']`` and the module
  declares ``REPLACES = ("scene_brief",)``.  ``scene_brief'`` is the ORIGINAL
  scene_brief object's schema / timeout / handler with ONLY the refusal
  sentence ("... say plainly that it is not something this robot can see ...")
  replaced by a pointer to look_for (§5.3): with the lane ON the old sentence
  would tell the model to refuse exactly the nouns look_for exists to learn.
* ``VocabularyManager`` (pure; no ROS, no I/O of its own) turns spoken nouns
  into ``(canonical label, prompt phrase)`` pairs, keeps the base vocabulary
  always on (a base noun is never sent as an extra), caps extras at 16,
  sends ONE ``/yolo/classes`` request per call carrying the FULL extras set
  (op ``set``, ``min_conf`` 0.35), coalesces calls inside 1.0 s of the last
  send, and NEVER short-circuits an identical set brain-side (the node already
  answers an unchanged list with ``swap_ms 0.0``; a brain-side cache was wrong
  after every node restart — critic B-M1).
* On the FIRST call of a process and on every ``mode="status"`` it sends
  ``op: query`` and re-seeds from the node's reply, so a brain restart does
  not forget what the node is looking for; a ``backend_fixed`` answer (the
  node runs the fixed-vocabulary engine) clears the extras brain-side.
* Replies always name the label to use ("label: bin (prompt: trash can) -
  say pick bin") and word confidence from RAW observation rows (scene_brief
  hides conf < 0.35): >= 0.35 "I see", 0.20-0.35 "might be ... not confident
  enough to pick", none "nothing matching yet".  Never "I see" without a row,
  never "applied" unless the gateway said ``DONE:applied`` (op set/reset) or
  ``DONE:state ... backend=yoloe`` (op query) WITH its ``extras=`` echo.
* Known limit (M8): look_for has NO fastpath row and the offline/keyword
  lane never reaches it.  The model lane is the only route: 'look for X' /
  'can you see X' reach the model exactly as with the lane OFF
  (brain/agent/fastpath.py is the base file, unchanged by batch 4), and the
  model calls look_for and relays its reply ('label: X - say pick X').

Label rules (D3): an EXTRA's wire label is a canonical single token
``^[a-z][a-z0-9_]{0,31}$`` ('bin', 'pencil_case'); the prompt phrase travels
separately.  An unseeded phrase keeps its FULL-phrase token ('widget holder'
-> 'widget_holder') — never a guessed shorter canonical (critic A-L3).

Config synonyms (``optional_tools.synonyms``): ONLY the nested-mapping form
with INLINE lists parses with brain/config_loader.py::

    optional_tools:
      open_vocab_lane: true
      synonyms:
        bin: [trash can, rubbish bin]

The block-list form (``bin:`` followed by ``- trash can`` lines) parses to
``{'bin': {}}`` and the flow-mapping form (``synonyms: {bin: [...]}``) parses
to a plain STRING — both are UNSUPPORTED: the manager prints a loud
``[open_vocab_lane]`` line and ignores the malformed entry (MEASURED with
config_loader._parse_text; pinned in tests/test_vocab_manager.py).  For a
canonical that the built-in SEED_TABLE already knows, config phrases ADD
synonyms and the built-in prompt is kept (the laptop-built embedding bank is
built from those prompts); for a NEW canonical the FIRST listed phrase is its
prompt.

Stdlib only, py3.8-safe.
"""

import difflib
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from brain import known_labels
from brain.observation import detection_base_class, label_base_class

#: scene_brief lane
REPLACES = ("scene_brief",)

LANE_TAG = "open_vocab_lane"

#: thresholds
#: extra admit floor = config.yaml det_conf_floor 0.35
#: GATE_CONF 0.20 = floor for publishing at all
CONFIDENT_CONF = 0.35
MIGHT_BE_CONF = 0.20

#: why 0.25
#: measured: 'bin' scored 0.35-0.39, 0.00-0.04 over
#: the old 0.35 floor, so any dip killed the row and
#: the object fell back to base labels (suitcase 0.55,
#: handbag 0.40). 0.25 leaves 0.10 margin; downstream
#: gates already let in >= 0.20. BASE stays 0.35
#: roll back: OPENBOT_EXTRA_CONF_FLOOR=0.35
EXTRA_CONF_DEFAULT = 0.25
EXTRA_CONF_RANGE = (0.20, 0.90)  # gateway range


def _extra_conf_floor(env: Any = None) -> float:
    """EXTRA admit floor: OPENBOT_EXTRA_CONF_FLOOR when it is a finite number
    inside EXTRA_CONF_RANGE, else EXTRA_CONF_DEFAULT."""
    raw = (os.environ if env is None else env).get("OPENBOT_EXTRA_CONF_FLOOR")
    if raw is None or str(raw).strip() == "":
        return EXTRA_CONF_DEFAULT
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return EXTRA_CONF_DEFAULT
    lo, hi = EXTRA_CONF_RANGE
    if v != v or not lo <= v <= hi:
        return EXTRA_CONF_DEFAULT
    return v


EXTRA_CONF_FLOOR = _extra_conf_floor()
MIN_CONF = EXTRA_CONF_FLOOR
MAX_EXTRAS = 236
MIN_INTERVAL_S = 1.0
SETTLE_S = 2.5
POLL_S = 0.25
#: call budget
#: under look_for.timeout_s (10.0), so the reply beats
#: the generic TIMEOUT envelope
CALL_BUDGET_S = 9.0
LOOK_FOR_TIMEOUT_S = 10.0

LABEL_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
PROMPT_RE = re.compile(r"^[a-z][a-z0-9 \-]{0,47}$")

#: base phrases
#: these NAME a base class, never sent as extras
#: (base vocab is always on); the reply still names
#: the label to use. 'wine glass' dropped from the
#: node allowlist
BASE_SYNONYMS: Dict[str, str] = {
    "mug": "cup", "coffee mug": "cup", "glass": "cup", "drinking glass": "cup",
    "teacup": "cup", "tea cup": "cup", "tumbler": "cup",
    "phone": "cell phone", "mobile": "cell phone",
    "mobile phone": "cell phone", "smartphone": "cell phone",
    "cellphone": "cell phone", "iphone": "cell phone",
    "plant": "potted plant", "house plant": "potted plant",
    "houseplant": "potted plant", "pot plant": "potted plant",
    "ball": "sports ball", "football": "sports ball",
    "soccer ball": "sports ball", "basketball": "sports ball",
    "tennis ball": "sports ball",
    "bag": "handbag", "purse": "handbag",
    "computer": "laptop", "notebook computer": "laptop",
    "water bottle": "bottle", "plastic bottle": "bottle",
    "remote control": "remote", "tv remote": "remote",
    "computer mouse": "mouse",
    "rucksack": "backpack", "schoolbag": "backpack", "school bag": "backpack",
    "luggage": "suitcase",
    "alarm clock": "clock", "wall clock": "clock",
    "flower vase": "vase",
    "textbook": "book", "novel": "book",
    "schoolbag": "backpack", "school bag": "backpack", "rucksack": "backpack",  # repeat
    "man": "person", "woman": "person", "people": "person",
}

#: the seed table
#: canonical -> (prompt, synonyms)
#: every prompt MUST be a line of ../assets/vocab_seed.txt
#: (the laptop bank builder's input)
#: or the node rejects it 'not_in_bank'
SEED_TABLE: Dict[str, Tuple[str, Tuple[str, ...]]] = OrderedDict([
    ("bin", ("trash can", ("bin", "trash can", "rubbish bin", "trash bin",
                           "garbage can", "garbage bin", "waste bin",
                           "wastebasket", "waste basket", "dustbin",
                           "litter bin", "trashcan"))),
    ("box", ("cardboard box", ("box", "cardboard box", "carton", "parcel",
                               "package"))),
    ("pencil_case", ("pencil case", ("pencil case", "pen case",
                                     "pencil pouch"))),
    ("basket", ("basket", ("basket", "wicker basket"))),
    ("tray", ("tray", ("tray", "serving tray"))),
    ("plate", ("plate", ("plate", "dinner plate", "dish"))),
    ("spoon", ("spoon", ("spoon", "teaspoon", "tablespoon"))),
    ("knife", ("knife", ("knife", "kitchen knife", "butter knife"))),
    ("scissors", ("scissors", ("scissors", "pair of scissors"))),
    ("pen", ("pen", ("pen", "ballpoint pen", "biro"))),
    ("pencil", ("pencil", ("pencil",))),
    ("marker", ("marker pen", ("marker", "marker pen", "felt tip pen",
                               "highlighter"))),
    ("notebook", ("notebook", ("notebook", "notepad", "exercise book"))),
    ("stapler", ("stapler", ("stapler",))),
    ("tape", ("roll of tape", ("tape", "roll of tape", "sticky tape",
                               "duct tape", "masking tape"))),
    ("keys", ("keys", ("keys", "key", "bunch of keys", "keychain",
                       "key ring"))),
    ("wallet", ("wallet", ("wallet",))),
    ("glasses", ("eyeglasses", ("glasses", "eyeglasses", "spectacles"))),
    ("headphones", ("headphones", ("headphones", "headset", "earphones"))),
    ("charger", ("phone charger", ("charger", "phone charger"))),
    ("cable", ("cable", ("cable", "usb cable", "charging cable", "wire"))),
    ("umbrella", ("umbrella", ("umbrella",))),
    ("hat", ("hat", ("hat", "cap", "baseball cap"))),
    ("shoe", ("shoe", ("shoe", "sneaker", "trainer", "boot"))),
    ("towel", ("towel", ("towel", "hand towel", "tea towel"))),
    ("sponge", ("sponge", ("sponge", "kitchen sponge"))),
    ("tissue_box", ("tissue box", ("tissue box", "box of tissues",
                                   "tissues", "kleenex"))),
    ("toilet_paper", ("toilet paper roll", ("toilet paper",
                                            "toilet paper roll",
                                            "toilet roll", "loo roll"))),
    ("can", ("drink can", ("can", "drink can", "soda can", "tin can",
                           "beer can", "coke can"))),
    ("jar", ("jar", ("jar", "glass jar"))),
    ("kettle", ("kettle", ("kettle", "electric kettle"))),
    ("pan", ("frying pan", ("pan", "frying pan", "skillet"))),
    ("pot", ("cooking pot", ("pot", "cooking pot", "saucepan"))),
    ("teddy_bear", ("teddy bear", ("teddy bear", "teddy", "stuffed animal",
                                   "plush toy", "soft toy"))),
    ("toy", ("toy", ("toy", "toy car", "toy block"))),
    ("bucket", ("bucket", ("bucket", "pail"))),
    ("lamp", ("lamp", ("lamp", "desk lamp", "table lamp"))),
    ("pillow", ("pillow", ("pillow", "cushion"))),
    ("toolbox", ("toolbox", ("toolbox", "tool box"))),
    ("screwdriver", ("screwdriver", ("screwdriver",))),
    ("hammer", ("hammer", ("hammer",))),
    ("lunchbox", ("lunch box", ("lunchbox", "lunch box"))),
    ("banana", ("banana", ("banana",))),
    ("orange", ("orange", ("orange", "orange fruit"))),
    ("toothbrush", ("toothbrush", ("toothbrush",))),
])

_ARTICLES = ("a", "an", "the", "some", "my", "your", "this", "that")
_SIZE_WORDS = ("small", "little", "tiny", "big", "large", "huge", "mini", "medium")
_COLOUR_WORDS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
                 "white", "black", "grey", "gray", "teal", "navy", "maroon")


def _strip_colour(phrase: str) -> tuple:
    """Extract leading colour word and return (colour|None, stripped_phrase)."""
    toks = phrase.split()
    if toks and toks[0] in _COLOUR_WORDS:
        return toks[0], " ".join(toks[1:])
    return None, phrase


def _norm_phrase(noun: Any, extract_colour: bool = False) -> Any:
    """Lowercase, punctuation -> space, collapse, strip leading articles and
    size words. If extract_colour is True, return (colour|None, phrase) tuple;
    otherwise return the normalised phrase (for backward compatibility)."""
    if not isinstance(noun, str):
        return (None, "") if extract_colour else ""
    text = re.sub(r"[^a-z0-9 \-]+", " ", noun.lower())
    toks = text.split()
    # leading articles
    while toks and toks[0] in _ARTICLES:
        toks.pop(0)
    # then size words
    while toks and toks[0] in _SIZE_WORDS:
        toks.pop(0)
    phrase = " ".join(toks)
    if extract_colour:
        colour, stripped = _strip_colour(phrase)
        return colour, stripped
    return phrase


def _singular_candidates(phrase: str) -> List[str]:
    """Lookup-only singular forms ('bins' -> 'bin', 'boxes' -> 'box').
    Used ONLY to find a phrase in an existing table, never to mint a label
    (so 'bus' is never shortened to 'bu')."""
    out = []
    if phrase.endswith("ies") and len(phrase) > 4:
        out.append(phrase[:-3] + "y")
    if phrase.endswith("es") and len(phrase) > 3:
        out.append(phrase[:-2])
    if phrase.endswith("s") and not phrase.endswith("ss") and len(phrase) > 2:
        out.append(phrase[:-1])
    return out
