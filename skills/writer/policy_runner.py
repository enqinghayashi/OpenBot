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
    def __init__(self, cfg, onnx_dir):
        self.action_scale = float(cfg["action_scale"])
        self.defaults = np.array(cfg["default_angles"], dtype=np.float32)  # 15
        self.gate_norm = float(cfg["gate_cmd_norm"])

        self._sessions = {}
        for key, fname in (("balance", cfg["policy_balance"]),
                           ("walk", cfg["policy_walk"])):
            path = os.path.join(onnx_dir, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"ONNX policy missing: {path}")
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            self._sessions[key] = sess
            self._in_name = sess.get_inputs()[0].name
            self._out_name = sess.get_outputs()[0].name

        # between ticks: last action (obs feedback), gate, targets
        self.action = np.zeros(15, dtype=np.float32)
        self.last_key = "balance"
        self.target_q = self.defaults.copy()

