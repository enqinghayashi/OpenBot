# brain/agent/optional_tools.py - stdlib only, runs on py3.8
"""Optional brain-tool lanes (batch 4 convention, SHARED with planner-sonic).

A lane is one row of ``OPTIONAL_TOOL_LANES``:
    (switch tag, module, factory name, the tool name the factory MUST return)

``ToolRegistry._register_defaults`` calls ``register_optional_tools(self,
self._optional_switches)`` exactly once, after the 13 default tools; the
switches dict is the ``optional_tools:`` config section, passed by
``voice/app.py build_brain_stack`` and ``G1Agent.from_config`` as
``cfg.get("optional_tools")``.

OFF-path contract (pinned by tests/test_optional_tools.py OFF-2): with the
section absent (None), empty, not a dict, or with every switch anything
other than the literal ``True``, nothing is imported and nothing is
registered — ``get_specs()`` is byte-identical to the base registry.

A lane is all-or-nothing: an import error, a factory error, a factory that
does not return its declared tool, or a name clash with an already
registered tool prints ONE loud line
    [<tag>] <module> FAILED (<exc>) - tool FORCED OFF
and leaves the registry exactly as it was (only what THIS lane added is
unregistered; any original it displaced via REPLACES is restored, by
identity).  It never raises — a broken optional lane can not stop the
brain from booting.

``SystemExit`` raised by a lane module (at import or inside its factory,
e.g. ``sys.exit()`` on a missing weights / text-encoder asset) is caught the
same way; ``KeyboardInterrupt`` still propagates (operator Ctrl-C).

Lane factories run INSIDE ``ToolRegistry.__init__`` — before
``voice/app.py`` calls ``registry.attach(scene_graph=...)`` and before
``G1Agent.from_config`` attaches its collaborators — so a lane tool must
read ``registry.scene_graph`` / ``registry.backend`` at CALL time, never
capture them in ``make_tools``.  ``switches`` must be a plain ``dict``
(``brain.config_loader`` returns one); any other Mapping reads as OFF.

A lane module may declare ``REPLACES = ('scene_brief',)`` to re-register an
existing tool under the same name (the ON-only scene_brief description,
design §5.3); the displaced original object is restored on rollback.

The fastpath / tool_contract / verifier rows of a lane remain that lane's
own job (a contract precondition must stay in the mapped set or
brain/tool_contract.py raises at import).
"""
import importlib

OPTIONAL_TOOL_LANES = (
    ("open_vocab_lane", "brain.look_for_tool", "make_tools", "look_for"),
    ("body_mode_lane",  "brain.body_mode_tool", "make_tools", "body_mode"),
    ("spatial_lane",    "brain.locate_tool",    "make_tools", "locate"),
)


