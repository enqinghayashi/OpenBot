#!/usr/bin/env python3
"""
Minimal G1 locomotion wrapper (TODO T1).

Wraps unitree_sdk2py.g1.loco.g1_loco_client.LocoClient plus the `sportmodestate`
DDS topic into a small class with one method per high-level locomotion verb.

Every method maps 1:1 to a real SDK call (see per-method docstrings). This module
NEVER writes leg joints -- the balance FSM stays in control (safety-critical,
see LOCO_PICKUP_BRIEF.md §2 / DETERMINISTIC_LOCO_PICKUP.md).

SDK layout (module-level constant, matches the existing loco_daemon.py style):
    - on robot:  /home/unitree/unitree_sdk2_python
    - on dev box: /home/leonlin/Unitree/Dependencies_Libraries/unitree_sdk2_python
"""

import json
import sys
import threading

# unitree_sdk2_python, the known-good copy. a bare import from the login
# shell can resolve to the broken xrkong master copy, whose rpc writes fail
# with "[ClientStub] send request error", so insert this path first.
UNITREE_SDK_PATH = "/home/unitree/unitree_sdk2_python"
sys.path.insert(0, UNITREE_SDK_PATH)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

# DDS topic with high-level sport-mode state (velocity / yaw_speed /
# position / mode / gait_type); sport_service publishes it.
SPORT_MODE_STATE_TOPIC = "rt/sportmodestate"

# get rpc ids that LocoClient.Init() registers (see
# unitree_sdk2py/g1/loco/g1_loco_api.py); get_fsm_id() and get_fsm_mode()
# use them.
LOCO_API_GET_FSM_ID = 7001
LOCO_API_GET_FSM_MODE = 7002


class G1Loco:
    """Minimal G1 locomotion wrapper.

    Owns a LocoClient (JSON-over-RPC to the `sport` service) and a
    `rt/sportmodestate` DDS subscription for read-only state feedback.
    """

    def __init__(self, net_interface: str = "eth0", state_capacity: int = 10):
        self._net_interface = net_interface
        self._state_capacity = state_capacity
        self._client = None  # built in connect(); SDK needs DDS up first
        self._state_sub = None
        self._state = None
        self._lock = threading.Lock()

    # -- lifecycle (connect / disconnect) ----------------------------------

