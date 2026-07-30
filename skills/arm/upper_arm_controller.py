#!/usr/bin/env python3
"""
upper_arm_controller.py — G1_29_ArmControllerUpperOnly (T4, Task 2)

Upper-only variant of G1_29_ArmController: the daemon streams arm joint
commands while the robot holds a pick (stationary, non-801), so this
controller drives only the two arms (15-28) and the arm_sdk enable-gate
joint 29 (kNotUsedJoint0).  The waist (12-14) AND the legs (0-11) are
locked (mode=1) at the captured pose and held — never driven.  Locking
the legs mirrors the stock G1_29_ArmController (all 35 joints mode=1),
which the arm_sdk gate requires: with the legs limp (mode=0) the gate is
not honored and the arms run limp (effective kp ~2 Nm/rad vs commanded).

Deployment: copy to Stuart's /home/unitree/leon/upper_arm_controller.py
(already on the daemon's sys.path).  Selected by the daemon argv flag
`upper` (pick_daemon_HORIZONTAL_version.py main()).

Differences vs G1_29_ArmController (robot_arm.py):
  * init locks arm joints 15-28 (mode=1, kp/kd per joint class), the
    gate 29, the waist 12-14 (mode=1, kp_low/kd_low, q seeded at the
    captured pose — held, not driven), the legs 0-11 (mode=1,
    kp_high/kd_high, q seeded at the captured pose — held, not driven),
    AND the trailing unused slots 30-34 (mode=1, kp_high/kd_high, q
    seeded at the captured pose — persist at the init value, never
    driven; the stock locks ALL 35 joints, and the G1 firmware trips if
    the trailing slots are mode=0 in Running Mode (801)).
  * the ctrl loop writes arm joints 15-28 + gate 29 + the locked
    waist 12-14 AND legs 0-11 (q held at the captured pose, dq/tau=0).
  * WRITE_WAIST = True (waist locked for the arm_sdk gate).
  * adds release_arms() / acquire_arms() (101-step gate ramps) and
    hold_only().

API-compatible with G1_29_ArmController: ctrl_dual_arm,
get_current_dual_arm_q, get_current_dual_arm_dq, ctrl_dual_arm_go_home,
speed_gradual_max, speed_instant_max, clip_arm_q_target,
get_current_motor_q, get_mode_machine.
"""

import numpy as np
import sys
import threading
import time

from unitree_sdk2py.core.channel import (ChannelPublisher, ChannelSubscriber,
                                         ChannelFactoryInitialize)  # dds imports
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (LowCmd_ as hg_LowCmd,
                                                    LowState_ as hg_LowState)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

# joint enums + lowstate from the stock controller
from teleop.robot_control.robot_arm import (G1_29_ArmController,
                                            G1_29_JointArmIndex,
                                            G1_29_JointIndex,
                                            G1_29_LowState,
                                            G1_29_Num_Motors,
                                            DataBuffer)

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

GATE_JOINT = G1_29_JointIndex.kNotUsedJoint0  # joint 29, arm_sdk gate
GATE_RAMP_STEPS = 101
GATE_RAMP_DT = 0.02

# waist 12-14 locked at the captured pose, NOT driven, so the arm_sdk
# gate (29) is honored -- same as the stock waist lock.  Legs 0-11 too:
# limp legs (mode=0) leave the gate unhonored and the arms limp
# (effective kp ~2 Nm/rad vs the commanded 80).
WAIST_JOINTS = (G1_29_JointIndex.kWaistYaw,
                G1_29_JointIndex.kWaistRoll,
                G1_29_JointIndex.kWaistPitch)

# legs 0-11 locked at the captured pose (mode=1, kp_high/kd_high), like
# the stock locking ALL 35.  Limp legs (mode=0) leave the gate unhonored
# and the arms at ~2 Nm/rad vs 80.  The pick is stationary (non-801).
LEG_JOINTS = (G1_29_JointIndex.kLeftHipPitch,
              G1_29_JointIndex.kLeftHipRoll,
              G1_29_JointIndex.kLeftHipYaw,
              G1_29_JointIndex.kLeftKnee,
              G1_29_JointIndex.kLeftAnklePitch,
              G1_29_JointIndex.kLeftAnkleRoll,
              G1_29_JointIndex.kRightHipPitch,
              G1_29_JointIndex.kRightHipRoll,
              G1_29_JointIndex.kRightHipYaw,
              G1_29_JointIndex.kRightKnee,
              G1_29_JointIndex.kRightAnklePitch,
              G1_29_JointIndex.kRightAnkleRoll)

# slots 30-34 locked too, like the stock locking ALL 35.  At mode=0 (the
# DDS default) they may trip the G1 firmware in Running Mode (801): arm
# oscillation / fsm_mode 0->3.  Held at the captured pose, never driven.
NOT_USED_JOINTS = (G1_29_JointIndex.kNotUsedJoint1,
                   G1_29_JointIndex.kNotUsedJoint2,
                   G1_29_JointIndex.kNotUsedJoint3,
                   G1_29_JointIndex.kNotUsedJoint4,
                   G1_29_JointIndex.kNotUsedJoint5)


class G1_29_ArmControllerUpperOnly(G1_29_ArmController):
    """Arms-driven variant: legs 0-11 + waist 12-14 + trailing 30-34 held.

    The arms (15-28) and the arm_sdk gate joint (29) are driven; the
    waist (12-14), legs (0-11), and trailing unused slots (30-34) are
    locked (mode=1) at the captured pose and held, never driven.  Locking
    mirrors the stock G1_29_ArmController (all 35 joints mode=1) so the
    arm_sdk gate is honored and the arms don't run limp.  Pick runs
    stationary (non-801).
    """

    def __init__(self, motion_mode=False, simulation_mode=False):
        logger_mp.info("Initialize G1_29_ArmControllerUpperOnly...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        # lock the waist too, or the gate (29) isn't honored; same for
        # the legs (limp = ~2 Nm/rad vs 80).
        self.WRITE_WAIST = True

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        # True once the OWNER sets a limit (set_velocity_limit /
        # speed_instant_max): the ramp below is then suppressed, so it
        # can't stomp the owner's cap (1.5 rad/s moves) back to
        # 20+10*min(1,t/5) every tick.  speed_gradual_max() takes it back.
        self._velocity_override = False

        # lowcmd pub, lowstate sub
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion,
                                                     hg_LowCmd)
        else:
            # debug topic, unused (it would fight loco), kept anyway for
            # symmetry with the stock controller.
            self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_29_ArmControllerUpperOnly] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_ArmControllerUpperOnly] Subscribe dds ok.")

        # hg lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        # snapshot the pristine lower-body fields (mode/q of 0-14) so the
        # ctrl loop can spot ANY accidental write, whatever the DDS
        # defaults are.
        self._lower_body_init = {
            id.value: (int(self.msg.motor_cmd[id].mode),
                       float(self.msg.motor_cmd[id].q))
            for id in G1_29_JointIndex if id.value < 15
        }

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")

        # ── lock the arms (15-28) + gate (29) ──
        # waist, legs and slots 30-34 below
        logger_mp.info("Lock arms + arm_sdk gate (waist + legs locked below)...\n")
        for id in G1_29_JointArmIndex:
            self.msg.motor_cmd[id].mode = 1
            if self._Is_wrist_motor(id):
                self.msg.motor_cmd[id].kp = self.kp_wrist
                self.msg.motor_cmd[id].kd = self.kd_wrist
            else:
                self.msg.motor_cmd[id].kp = self.kp_low
                self.msg.motor_cmd[id].kd = self.kd_low
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        # gate 29: mode 1 (arm_sdk enable/weight);
        # q = 1.0 in motion mode (the ctrl loop sets it)
        self.msg.motor_cmd[GATE_JOINT].mode = 1
        self.msg.motor_cmd[GATE_JOINT].kp = self.kp_high
        self.msg.motor_cmd[GATE_JOINT].kd = self.kd_high
        self.msg.motor_cmd[GATE_JOINT].q = 1.0 if self.motion_mode else 0.0
        logger_mp.info("Arms locked OK!\n")

        # ── waist 12-14 ──
        # limp waist = gate (29) not honored ("arm barely moves").
        # Lock at the captured pose, HOLD it, never drive it.
        logger_mp.info("Lock waist (12-14) at captured pose...\n")
        self._waist_q_captured = np.array(
            [self.all_motor_q[id] for id in WAIST_JOINTS])
        for idx, id in enumerate(WAIST_JOINTS):
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp_high  # jitter fix: was kp_low=80
            self.msg.motor_cmd[id].kd = self.kd_low
            self.msg.motor_cmd[id].q = self._waist_q_captured[idx]
        logger_mp.info("Waist locked OK!\n")

        # ── legs 0-11 ──
        # same as the waist: captured pose, HOLD, never drive.  Limp legs
        # leave effective kp at ~2 Nm/rad vs the commanded 80.
        logger_mp.info("Lock legs (0-11) at captured pose...\n")
        self._leg_q_captured = np.array(
            [self.all_motor_q[id] for id in LEG_JOINTS])
        for idx, id in enumerate(LEG_JOINTS):
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp_high
            self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q = self._leg_q_captured[idx]
        logger_mp.info("Legs locked OK!\n")

        # ── slots 30-34 ──
        # init value only: the ctrl loop never writes them.  At mode=0
        # they may trip the firmware in Running Mode (801) -- see above.
        logger_mp.info("Lock trailing unused joints (30-34) at captured pose...\n")
        for id in NOT_USED_JOINTS:
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp_high
            self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        logger_mp.info("Trailing unused joints locked OK!\n")

        # startup debug: gate q + mode bits; check every group is locked
        # before the thread runs.
        gate_q = float(self.msg.motor_cmd[GATE_JOINT].q)
        leg_modes = [self.msg.motor_cmd[id].mode for id in range(12)]
        waist_modes = [self.msg.motor_cmd[id].mode for id in range(12, 15)]
        arm_modes = [self.msg.motor_cmd[id].mode for id in range(15, 29)]
        not_used_modes = [self.msg.motor_cmd[id].mode for id in range(29, 35)]
        sys.stderr.write(
            f"[upper] DEBUG gate_q={gate_q:.4f} "
            f"leg_modes={leg_modes} waist_modes={waist_modes} "
            f"arm_modes={arm_modes} not_used_modes={not_used_modes}\n")

        # publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_29_ArmControllerUpperOnly OK!\n")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def _ctrl_motor_state(self):
        """250 Hz loop writing the arms (15-28) + gate (29) + the locked
        waist (12-14) AND legs (0-11) held at the captured pose, CRC before
        every Write."""
        if self.motion_mode:
            self.msg.motor_cmd[GATE_JOINT].q = 1.0

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                clipped_arm_q_target = arm_q_target
            else:
                clipped_arm_q_target = self.clip_arm_q_target(
                    arm_q_target, velocity_limit=self.arm_velocity_limit)

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = clipped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            if self.WRITE_WAIST:
                # hold the waist each tick (mode=1 at init);
                # do NOT drive it
                for idx, id in enumerate(WAIST_JOINTS):
                    self.msg.motor_cmd[id].q = self._waist_q_captured[idx]
                    self.msg.motor_cmd[id].dq = 0
                    self.msg.motor_cmd[id].tau = 0

            # hold the legs each tick (mode=1 at init);
            # do NOT drive them
            for idx, id in enumerate(LEG_JOINTS):
                self.msg.motor_cmd[id].q = self._leg_q_captured[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = 0

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            # ramp only while the OWNER hasn't set a limit: else it
            # overwrites the daemon's 1.5 rad/s every tick and silently
            # lifts the cap to 30 rad/s within 5 s.
            if self._speed_gradual_max is True and not self._velocity_override:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    # ── gate ramp ──
    def release_arms(self):
        """Ramp the arm_sdk gate from 1 -> 0 in 101 steps (~2 s).

        The locomotion stack's arm fallback (arm_sdk disabled) takes the
        arms while the robot walks; after this returns the daemon must not
        stream arm targets until acquire_arms()."""
        logger_mp.info("[G1_29_ArmControllerUpperOnly] release_arms: "
                       "gate 1 -> 0")
        for w in np.linspace(1, 0, num=GATE_RAMP_STEPS):
            self.msg.motor_cmd[GATE_JOINT].q = float(w)
            time.sleep(GATE_RAMP_DT)
        logger_mp.info("[G1_29_ArmControllerUpperOnly] arms released")

    def acquire_arms(self):
        """Ramp the arm_sdk gate from 0 -> 1 in 101 steps (~2 s).

        After this the daemon resumes full authority over the arms."""
        logger_mp.info("[G1_29_ArmControllerUpperOnly] acquire_arms: "
                       "gate 0 -> 1")
        for w in np.linspace(0, 1, num=GATE_RAMP_STEPS):
            self.msg.motor_cmd[GATE_JOINT].q = float(w)
            time.sleep(GATE_RAMP_DT)
        logger_mp.info("[G1_29_ArmControllerUpperOnly] arms acquired")

    def hold_only(self):
        """Freeze arm targets at the current joint angles (gate stays on).

        The ctrl loop keeps streaming, but the target never moves away
        from the measured q, so the arms hold the current posture while
        the robot continues walking."""
        with self.ctrl_lock:
            self.q_target = self.get_current_dual_arm_q()
        logger_mp.info("[G1_29_ArmControllerUpperOnly] hold_only: targets "
                       "frozen at current q")

    # ── stock API ──
    def ctrl_dual_arm(self, q_target, tauff_target):
        """Set control target values q & tau of the left and right arm motors."""
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """Return current state q of all body motors (35)."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q
                         for id in G1_29_JointIndex])

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q
                         for id in G1_29_JointArmIndex])

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq
                         for id in G1_29_JointArmIndex])

    def ctrl_dual_arm_go_home(self):
        """Move both arms to home (q=0) — arms-only version."""
        logger_mp.info("[G1_29_ArmControllerUpperOnly] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
        tolerance = 0.05
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[G1_29_ArmControllerUpperOnly] both arms reached home.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        """Parameter t is the total time required for arms velocity to
        gradually increase to its maximum value, in seconds. Default 5.0."""
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True
        # stock ramp takes the limit last; an owner set wins
        self._velocity_override = False

    def speed_instant_max(self):
        """set arms velocity to the maximum value immediately."""
        self.arm_velocity_limit = 30.0
        # no ramp for an owner set
        self._velocity_override = True

    def set_velocity_limit(self, limit):
        """T8: explicit OWNER-set velocity cap (e.g. the pick daemon's
        1.5 rad/s slow moves / 2.0 rad/s go-home).

        Takes precedence over the stock gradual ramp: _ctrl_motor_state
        would otherwise overwrite this value with 20+10*min(1,t/5) on every
        tick while speed_gradual_max is active, silently lifting the cap to
        30 rad/s within 5 s (Task 8 diagnosis — the daemon's cap was dead).
        """
        self.arm_velocity_limit = float(limit)
        self._velocity_override = True

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        clipped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return clipped_arm_q_target

