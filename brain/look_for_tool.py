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


def _mint_label(phrase: str) -> str:
    """The canonical token of an UNSEEDED phrase: the FULL phrase, never a
    shortened guess ('widget holder' -> 'widget_holder')."""
    return re.sub(r"[^a-z0-9]+", "_", phrase).strip("_")


# =============================================================================
# VocabularyManager - pure state, canonicalisation, and the coalescing sender
# =============================================================================


class Canon(object):
    """Result of canonicalising one spoken noun."""

    __slots__ = ("kind", "label", "prompt", "phrase")

    def __init__(self, kind: str, label: str, prompt: Optional[str],
                 phrase: str) -> None:
        self.kind = kind  # base/extra/invalid
        self.label = label  # base or canonical
        self.prompt = prompt  # None for base
        self.phrase = phrase  # normalised phrase

    def as_tuple(self) -> Tuple[str, Optional[str]]:
        return (self.label, self.prompt)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Canon(%r, %r, %r, %r)" % (self.kind, self.label, self.prompt,
                                          self.phrase)


class VocabularyManager(object):
    """Brain-side open-vocabulary state.  No ROS; time and sleep injectable.

    ``desired``: canonical -> prompt, what the brain asks perception to look
    for (<= MAX_EXTRAS).  ``active``: the subset the node CONFIRMED
    (``DONE:applied``); only confirmed labels are registered in
    ``known_labels`` so pick/navigate never resolve an unconfirmed noun.
    """

    def __init__(self, synonyms: Any = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 min_interval_s: float = MIN_INTERVAL_S,
                 max_extras: int = MAX_EXTRAS,
                 min_conf: float = MIN_CONF,
                 log: Callable[[str], None] = print) -> None:
        self._clock = clock
        self._sleep = sleep
        self.min_interval_s = float(min_interval_s)
        self.max_extras = int(max_extras)
        self.min_conf = float(min_conf)
        self._log = log
        # canonical -> prompt
        self.prompts: Dict[str, str] = OrderedDict()
        self.phrase_to_label: Dict[str, str] = {}
        self.vocab_prompts: Dict[str, str] = {}  # prompt -> label
        for label, (prompt, phrases) in SEED_TABLE.items():
            self._learn(label, prompt, phrases)
        self._load_vocab_prompts()
        self._seed_from_config(synonyms)
        self.desired: Dict[str, str] = OrderedDict()
        self.active: Dict[str, str] = OrderedDict()
        self.queried = False
        # sender state
        self._cv = threading.Condition()
        self._gen = 0
        self._sent_gen = 0
        self._sending = False
        self._last_send_t: Optional[float] = None
        self._last_result: Any = None

    # -- seed -------------------------------------------------------------------

    def _learn(self, label: str, prompt: str, phrases: Any) -> None:
        self.prompts.setdefault(label, prompt)
        self.phrase_to_label[label.replace("_", " ")] = label
        self.phrase_to_label[prompt] = label
        for p in phrases:
            norm = _norm_phrase(p, extract_colour=False)
            if norm:
                self.phrase_to_label[norm] = label

    def _load_vocab_prompts(self) -> None:
        """Load the vocabulary prompts file if it exists."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vocab_file = os.path.join(repo_root, "brain", "vocab_prompts.txt")
        if os.path.isfile(vocab_file):
            try:
                with open(vocab_file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            self.vocab_prompts[line] = line
            except Exception:  # noqa: BLE001
                pass

    def _seed_from_config(self, synonyms: Any) -> None:
        if synonyms is None:
            return
        if not isinstance(synonyms, Mapping):
            self._log("[%s] optional_tools.synonyms ignored: expected a mapping "
                      "with inline lists (bin: [trash can, rubbish bin]), got "
                      "%s %r" % (LANE_TAG, type(synonyms).__name__, synonyms))
            return
        for raw_label, phrases in synonyms.items():
            label = _mint_label(_norm_phrase(str(raw_label), extract_colour=False))
            if (not isinstance(phrases, list) or not phrases
                    or not all(isinstance(p, str) for p in phrases)):
                self._log("[%s] optional_tools.synonyms.%s ignored: expected an "
                          "inline list of phrases ([a, b]), got %r (the block-list "
                          "form is unsupported)" % (LANE_TAG, raw_label, phrases))
                continue
            norm = [_norm_phrase(p, extract_colour=False) for p in phrases]
            norm = [p for p in norm if p]
            if (not LABEL_RE.match(label) or label in known_labels.TRACKED_CLASSES
                    or not norm or not all(PROMPT_RE.match(p) for p in norm)
                    or any(self._base_of(p) for p in norm)):
                self._log("[%s] optional_tools.synonyms.%s ignored: invalid label "
                          "or phrase (or a phrase that names a base class)"
                          % (LANE_TAG, raw_label))
                continue
            self._learn(label, self.prompts.get(label, norm[0]), norm)

    # -- canonicalisation --------------------------------------------------------

    @staticmethod
    def _base_of(phrase: str) -> Optional[str]:
        """The base TRACKED class a phrase names, or None."""
        tracked = known_labels.TRACKED_CLASSES
        if phrase in tracked:
            return phrase
        if phrase in BASE_SYNONYMS:
            return BASE_SYNONYMS[phrase]
        alias = known_labels.TARGET_ALIASES.get(phrase)
        if alias in tracked:
            return alias
        stripped = label_base_class(phrase)  # colour prefix
        if stripped in tracked:
            return stripped
        if stripped in BASE_SYNONYMS:
            return BASE_SYNONYMS[stripped]
        return None

    def canon(self, noun: Any) -> Canon:
        colour, phrase = _norm_phrase(noun, extract_colour=True)
        if not phrase:
            return Canon("invalid", "", None, (colour or "") + (" " if colour else "") + phrase)
        # colour is perception's decoration
        core = label_base_class(phrase) or phrase
        # lookup: synonyms, prompts, stripped, singular, last words
        lookup_candidates = [core] + _singular_candidates(core) + self._last_n_words(core, 2) + self._last_n_words(core, 1)
        for cand in lookup_candidates:
            if not cand:
                continue
            base = self._base_of(cand)
            if base is not None:
                return Canon("base", base, None, phrase)
            label = self.phrase_to_label.get(cand)
            if label is not None:
                return Canon("extra", label, self.prompts[label], phrase)
        label = _mint_label(core)
        if not LABEL_RE.match(label) or not PROMPT_RE.match(core):
            return Canon("invalid", label, None, phrase)
        return Canon("extra", label, core, phrase)

    def _last_n_words(self, phrase: str, n: int) -> List[str]:
        """Return list of last n words from phrase."""
        toks = phrase.split()
        if len(toks) <= n:
            return []
        return [" ".join(toks[-n:])]

    # -- known_labels mirror -----------------------------------------------------

    def _aliases_for(self, labels: Any) -> Dict[str, str]:
        wanted = set(labels)
        return {p: l for p, l in self.phrase_to_label.items()
                if l in wanted and p != l}

    def set_active(self, labels: Any) -> None:
        """The node's confirmed extras become the brain's active set and the
        ONLY extras known_labels resolves."""
        labels = [l for l in labels
                  if LABEL_RE.match(l) and l not in known_labels.TRACKED_CLASSES]
        self.active = OrderedDict(
            (l, self.desired.get(l) or self.prompts.get(l) or l.replace("_", " "))
            for l in labels)
        known_labels.clear_extra_classes()
        if self.active:
            known_labels.register_extra_classes(
                list(self.active), aliases=self._aliases_for(self.active))

    def clear(self) -> None:
        self.desired = OrderedDict()
        self.set_active([])

    def seed_from_node(self, labels: Any) -> None:
        """op:query answer: the node's extras become desired AND active."""
        clean = [l for l in labels
                 if LABEL_RE.match(l) and l not in known_labels.TRACKED_CLASSES]
        self.desired = OrderedDict(
            (l, self.prompts.get(l) or l.replace("_", " ")) for l in clean)
        for l, p in self.desired.items():
            self.phrase_to_label.setdefault(p, l)
            self.prompts.setdefault(l, p)
        self.set_active(clean)

    # -- message -----------------------------------------------------------------

    def message(self) -> Dict[str, Any]:
        """ONE request carrying the FULL extras set (idempotent replace)."""
        if not self.desired:
            return {"op": "reset", "classes": []}
        return {"op": "set",
                "classes": [{"label": l, "prompt": p}
                            for l, p in self.desired.items()],
                "min_conf": self.min_conf}

    # -- coalescing sender ---------------------------------------------------------

    def mark_dirty(self) -> int:
        """Record that ``desired`` changed (or was re-requested); returns the
        generation the caller must see sent."""
        with self._cv:
            self._gen += 1
            return self._gen

    def send(self, transmit: Callable[[Dict[str, Any]], Any], gen: int) -> Any:
        """Send the FULL set once generation ``gen`` is covered.

        A call inside ``min_interval_s`` of the last send waits for the window
        to close; every generation recorded while it waited rides on the same
        single send (coalescing).  A caller whose generation was already
        covered by another thread's send returns that send's result without
        transmitting.  There is deliberately NO 'set unchanged -> skip'
        short-circuit (critic B-M1).
        """
        while True:
            with self._cv:
                if self._sent_gen >= gen:
                    return self._last_result
                if self._sending:
                    self._cv.wait(0.05)
                    continue
                wait_s = 0.0
                if self._last_send_t is not None:
                    wait_s = self._last_send_t + self.min_interval_s - self._clock()
                if wait_s <= 0:
                    self._sending = True
                    covered = self._gen
                    msg = self.message()
                    break
            self._sleep(min(wait_s, 0.05))
        result = None
        try:
            result = transmit(msg)
        finally:
            with self._cv:
                self._sending = False
                self._sent_gen = covered
                self._last_send_t = self._clock()
                self._last_result = result
                self._cv.notify_all()
        return result


# =============================================================================
# making sense of the gateway reply
# =============================================================================

_EXTRAS_RE = re.compile(r"\bextras=([a-z0-9_,]*)")
_REJECTED_RE = re.compile(r"\brejected=(\S*)")
_BACKEND_RE = re.compile(r"\bbackend=(\S+)")
_REFUSAL_RE = re.compile(r"ERROR:\s*yolo_classes\s+([a-z_]+)")
_REFUSAL_KINDS = {"backend_fixed": "backend_fixed", "busy": "busy",
                  "disabled": "disabled", "swap_failed": "swap_failed",
                  "bad_request": "bad_request"}


def _env_text(env: Any) -> Tuple[Optional[bool], str, Any]:
    """(success, text, raw value) from an envelope of either shape:
    {'success', 'value'|'reason'} (brain envelope) or {'ok', 'text'}."""
    if not isinstance(env, dict):
        return None, str(env), env
    success = env.get("success", env.get("ok"))
    value = env.get("value")
    if value is None:
        value = env.get("text", env.get("reply"))
    if success is False or value is None:
        text = env.get("reason") or value or ""
    else:
        text = value
    return success, text if isinstance(text, str) else str(text), value


class GatewayOutcome(object):
    """Classified result of one yolo_classes round-trip."""

    __slots__ = ("kind", "text", "extras", "rejected")

    def __init__(self, kind: str, text: str, extras: Optional[List[str]] = None,
                 rejected: Optional[Dict[str, str]] = None) -> None:
        self.kind = kind  # applied|unconfirmed|backend_fixed|busy|
        self.text = text  # disabled|swap_failed|bad_request|no_verb
        self.extras = extras  # confirmed extras
        self.rejected = rejected or {}
