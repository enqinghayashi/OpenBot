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

    def connect(self):
        """Bring up DDS + the sport client and subscribe to sportmodestate.

        Calls ChannelFactoryInitialize(0, net_interface), then constructs a
        LocoClient and calls LocoClient.SetTimeout(10.0), LocoClient.Init()
        (registers the loco RPC API ids) and LocoClient.WaitLeaseApplied(), then
        subscribes to `rt/sportmodestate` via ChannelSubscriber.Init(handler, queueLen).
        """
        ChannelFactoryInitialize(0, self._net_interface)
        client = self._client = LocoClient()
        client.SetTimeout(10.0)
        client.Init()
        client.WaitLeaseApplied()

        sub = ChannelSubscriber(SPORT_MODE_STATE_TOPIC, SportModeState_)
        sub.Init(self._on_state, self._state_capacity)
        self._state_sub = sub
        return self

    def disconnect(self):
        """Release the sportmodestate DDS subscription.

        Closes the `rt/sportmodestate` ChannelSubscriber via
        ChannelSubscriber.Close(). LocoClient has no explicit RPC disconnect;
        call stop() before this to halt motion (damping is a caller choice).
        """
        with self._lock:
            sub, self._state_sub = self._state_sub, None
            self._state = None
        if sub is not None:
            sub.Close()

    # -- locomotion verbs (one per SDK call) --------------------------------

    @property
    def _loco(self) -> LocoClient:
        """Resolved LocoClient; raises if connect() has not been called."""
        if self._client is None:
            raise RuntimeError("G1Loco not connected: call connect() first")
        return self._client

    def set_velocity(self, vx: float, vy: float, omega: float, duration: float = 1.0):
        """Command default G1 walking at (vx forward, vy lateral, omega yaw).

        Maps to LocoClient.SetVelocity(vx, vy, omega, duration) (RPC 7105). The
        balance FSM executes the gait; no leg joints are written here.
        """
        return self._loco.SetVelocity(vx, vy, omega, duration)

    def start(self):
        """Call LocoClient.Start(). HARDWARE-DAY FINDING (2026-08-15): the FSM
        id is COPY-DEPENDENT — Stuart's SDK copy sends SetFsmId(200), the
        dev-box/upstream copies send 500 (walk-node/hardware-day/
        RECOMMENDATION.md §2(a)). From R2+A the robot is already in sport-FSM
        801, so walk_node/sweep_test deliberately SKIP this call on hardware;
        the verb stays for API compatibility. No leg joints are written.
        """
        return self._loco.Start()

    def stop(self):
        """Halt locomotion: maps to LocoClient.StopMove() == SetVelocity(0, 0, 0)."""
        return self._loco.StopMove()

