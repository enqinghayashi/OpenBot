"""
brain/reliability.py — the quantitative E2E reliability model.

Stdlib only.  Port of the harness reliability model (Thea SS2 framing in
docs/design_rationale.md): an open-loop agent succeeds with the PRODUCT of
per-step probabilities; a harnessed agent runs a verifier after each step
and only proceeds on a confirm signal, so each step's contribution becomes
its posterior success probability GIVEN the verifier confirmed.

SHARED CONTRACT (exact names):
    open_loop(p_i: list|float)
    harness(p_i, chi, P)
    per-step ceiling(p, chi) = p*chi / (p*chi + (1-p)*(1-chi))

The per-step ceiling is the Bayes posterior of "step actually succeeded"
given a verifier confirm, under a symmetric reliability model
(sensitivity == specificity == chi):

    P(success | confirm) = chi*p / (chi*p + (1-chi)*(1-p))

Derived properties (verified algebraically in the smoke test):
    - chi = 0.5  -> ceiling == p exactly (no-gain; a coin-flip verifier
                    adds nothing) -> harness == open_loop of the full chain.
    - chi = 1.0  -> ceiling == 1.0 for any p > 0 (perfect verifier: every
                    confirmed step is certain) -> the chain's residual
                    failure compresses to the unverified base term (1 - P).
    - ceiling >= p  iff  chi >= 0.5 (the verifier only helps above chance).
    - Denominator-0 corners are SYMMETRIC and both return 0.0:
      (chi==1, p==0) and (chi==0, p==1) each condition on the confirm of
      an impossible event (the posterior is an undefined 0/0); a step
      that never succeeds, or a verifier that never confirms, cannot
      yield a confirmed success.

`P` is the probability of the UNVERIFIED base condition — the thing that
must hold before any step can succeed, e.g. the goal object's presence
(a natural choice: the target detection's confidence).  It is not subject
to verification, so it multiplies the chain untouched:

    harness(p_i, chi, P) = P * prod(ceiling(p, chi) for p in p_i)

Usage: substitute measured per-step p_i and verifier chi to predict E2E
success before (open_loop) vs after (harness) — the thesis quantitative
result.  Inputs are clamped into [0, 1] (float noise must not break the
model); clamping is documented, not silent magic.
"""

from __future__ import annotations

from typing import List, Union

Number = Union[int, float]


def _clamp(value: float, name: str) -> float:
    if value < 0.0 or value > 1.0:
        value = min(1.0, max(0.0, value))
    return value


def ceiling(p: Number, chi: Number) -> float:
    """Per-step posterior: P(step succeeded | verifier confirmed).

    Bayesian update under a symmetric verifier of reliability `chi`.
    chi = 0.5 returns p exactly (no-gain); chi = 1 returns 1.0 for p > 0
    (exponential compression of the per-step failure term).
    """
    p = _clamp(float(p), "p")
    chi = _clamp(float(chi), "chi")
    if chi == 0.5:
        return p  # exact here, no float dust
    denom = p * chi + (1.0 - p) * (1.0 - chi)
    if denom <= 0.0:
        # measure-zero corners, symmetric: (chi==1, p==0), (chi==0, p==1)
        # both condition on an impossible event - 0/0 posterior.
        # 0.0 either way: no step success, no confirmed success.
        return 0.0
    return p * chi / denom
