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


def switch_on(switches, key):
    """True ONLY when ``switches`` is a plain ``dict`` and ``switches.get(key)``
    is the literal ``True`` (a string 'true', 1, or any other truthy value
    reads OFF — the same rule ``register_optional_tools`` applies to a lane).

    Used for the batch-4 non-lane switches that live in the SAME hand-merged
    ``optional_tools:`` section (architecture §17 M6):
    ``hint_charset_refusal`` and ``target_disambiguation``.  Non-lane keys
    are ignored by ``register_optional_tools`` (it iterates only
    ``OPTIONAL_TOOL_LANES``), so they register no tool and leave
    ``get_specs()`` byte-identical."""
    return isinstance(switches, dict) and switches.get(key) is True


def register_optional_tools(registry, switches, importer=importlib.import_module):
    """All-or-nothing per lane. switch must be exactly True. Any failure
    (import, factory, name clash) -> print('[<tag>] <mod> FAILED (<exc>) - tool FORCED OFF')
    and skip the lane; never raise. A lane module may declare REPLACES = ('scene_brief',)
    to re-register an existing tool under the same name (restored on rollback).

    Returns the list of tool names registered (or re-registered) by the
    lanes that went ON, in lane order."""
    if not isinstance(switches, dict):
        return []
    registered = []
    for tag, mod_name, factory_name, tool_name in OPTIONAL_TOOL_LANES:
        if switches.get(tag) is not True:
            continue
        done = []        # this lane's adds
        replaced = []    # old Tool objects
        order = None     # pre-swap order
        try:
            mod = importer(mod_name)
            tools = list(getattr(mod, factory_name)(registry))
            names = [t.name for t in tools]
            if tool_name not in names:
                raise RuntimeError("factory returned %r, expected %r" % (names, tool_name))
            allowed = tuple(getattr(mod, "REPLACES", ()))
            for t in tools:
                if registry.has(t.name) and t.name not in allowed:
                    raise RuntimeError("name clash: %r" % t.name)
            for t in tools:
                # `not in done`: a dup inside one factory list must clash,
                # not push out this lane's own tool
                if t.name in allowed and registry.has(t.name) and t.name not in done:
                    if order is None:
                        order = list(registry.snapshot())
                    replaced.append(registry.snapshot()[t.name])
                    registry.unregister(t.name)
                registry.register(t)
                done.append(t.name)
            registered.extend(names)
        except (Exception, SystemExit) as exc:  # off, never crash
            for n in done:
                registry.unregister(n)          # this lane only
            for orig in replaced:
                registry.register(orig)         # originals back
            if replaced:
                _restore_order(registry, order)  # get_specs() order as well
            print("[%s] %s FAILED (%s) - tool FORCED OFF" % (tag, mod_name, exc))
    return registered
