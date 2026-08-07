"""walk_lock.py -- B8l pure laws for the walk FSM (stdlib only, py3.8).

Ships co-located with walk_node.py (robot: openbot/walk_lock.py; repo:
skills/locomotion/walk_lock.py).  No rclpy, no numpy: unit-testable.

F1  ApproachLock -- a BARE-label walk ('bin') keeps the ONE object it first
    saw.  MEASURED 2026-10-16 (walk.log 15:50:03-08, depth.log 14757-14781):
    the walk latched a bin at x -0.33 y 1.00 z -0.50, then followed rows
    labelled 'bin' at z -0.81 (y 1.36) and z +0.10 (y 0.42) -- other things --
    and its odometry-world target wandered 1.2 m while Stuart only turned.
    The walk instance lock (B8, '<label>#<iid>') only existed for targets the
    brain named by id.  The lock follows the object by tracker id ACROSS label
    changes (the tracker associates class-agnostically; on 10-16 the same bin
    track went bin -> backpack -> suitcase), and by geometry when the id dies
    (ids churned 246 -> 251 -> 253 in 10 s with the robot standing still,
    gateway.log [lock]).  A row of the target label ELSEWHERE is not the target.

F5  reacquire_turn -- after losing the target, turn back toward where it was
    (odometry when fresh, else the last body-frame bearing) instead of
    sweeping one fixed way.

F7  pose_command / pose_blocked -- walk to an odometry pose (the start point
    of a fetch) and face the requester.
"""
import math

# ── frame helpers we both use ────────────────────────────────────────────────
# detection body frame has x = RIGHT, y = FORWARD (walk_node, depth projector).
# odometry (/unitree/slam_mapping/odom, REP-103) is x = forward, y = left, yaw
# CCW (+omega = LEFT), so mind the swap.


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def body_to_world(pose, x_b, y_b):
    """Body-frame (right, forward) point -> odometry world (x, y)."""
    rx, ry, yaw = pose[0], pose[1], pose[2]
    fwd, left = y_b, -x_b
    return (rx + math.cos(yaw) * fwd - math.sin(yaw) * left,
            ry + math.sin(yaw) * fwd + math.cos(yaw) * left)


def world_to_body(pose, wx, wy):
    """Odometry world (x, y) -> body frame (right, forward)."""
    rx, ry, yaw = pose[0], pose[1], pose[2]
    dx, dy = wx - rx, wy - ry
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return -left, fwd


def bearing(x_b, y_b):
    """walk_node's yaw error: atan2(x, y); positive = target to the RIGHT."""
    return wrap_pi(math.atan2(x_b, y_b))
