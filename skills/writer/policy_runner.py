"""policy_runner.py — GR00T Balance/Walk ONNX sessions with the cmd gate.

Port of g1_gear_wbc_policy.py: gate on the RAW rc command norm
(norm(cmd) < 0.05 => Balance policy_1, else Walk policy_2), action transform
target_q = action * 0.25 + default_angles[0:15]. Inference is CPU onnxruntime;
the 50 Hz tick is called inline from the writer loop (never a worker thread —
GIL lesson). ~1-4 ms per inference, targets hold constant during that tick.
"""

import os

import numpy as np
import onnxruntime as ort


class PolicyRunner:
