#!/usr/bin/env python3
"""Direct sensitivity probe: does the Balance ONNX's hip-roll OUTPUT depend on
the hip-roll OBSERVATION?

WHY THIS EXISTS
---------------
On the robot, `stance 4` returned DONE, the socket reported `stance=+4.0`, and
the log proves `_write_stance` rebound BOTH arrays (:922-923 run before the
`_stance_log` call that printed).  Code reading says the observation therefore
carries `-0.0698` on the hip-roll entries and the readout carries `+0.0698`.
The measured hip roll moved **0.0 deg**.

A zero that exact needs one of exactly two explanations:
  (a) the policy is DEAF to the hip-roll obs entries -> the whole stance design
      is dead by construction, regardless of the robot; or
  (b) the policy is NOT deaf, and the box's chain is broken somewhere a
      --verbose restart would reveal.

This script settles (a) vs (b) with the writer's OWN classes and the REAL ONNX,
on the laptop, touching nothing.

WHAT IT MEASURES
----------------
At a fixed operating point it reports, per commanded degree of dual shift:
  shift-obs      -- perturb obs.default_padded only (what the policy SEES)
  shift-policy   -- perturb policy.defaults only  (the readout offset)
  shift-both     -- both, exactly as _write_stance does (THE SHIPPED ONE)

and the raw partial derivative d(target_q[1]) / d(obs_frame[14]).
"""
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WRITER = os.path.dirname(HERE)
sys.path.insert(0, WRITER)

from obs_builder import ObsBuilder          # noqa: E402
from policy_runner import PolicyRunner      # noqa: E402

CFG = os.path.join(WRITER, "config", "merge_config.yaml")
DEG = 4.0
D = DEG * np.pi / 180.0
# _write_stance touches: hip L(+), R(-),
# ankle L(+), R(-); ank = STANCE_ANKLE_RATIO*hip
ANK_RATIO = -0.85
IDX = {1: +D, 7: -D, 5: +ANK_RATIO * D, 11: -ANK_RATIO * D}


