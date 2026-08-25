"""PcConfig: every threshold the point-cloud consumer uses, in one place.

Python 3.8-safe (the box runs ROS 2 Foxy; py3.8 is INFERRED).  stdlib only.

Each field carries a label in ``LABELS``:
  MEASURED  - read from a file / log, or computed by running code
  DERIVED   - arithmetic on MEASURED values
  INFERRED  - rests on a stated assumption; REPLACE from a box capture
  POLICY    - a design choice (a gate, a bound), not a physical quantity

Calibration (docs/POINTCLOUD_CONSUMER_DESIGN.md section 10) replaces INFERRED
defaults with MEASURED values from capture_clouds.py output.  A calibrated value
must carry its capture filename in LABELS.
"""
import dataclasses
import hashlib


