"""gateway package — brain<->skills bridge on the G1 ("Stuart").

Holds two layers:

* ``assemble`` (pure stdlib): raw skill payload dicts (detections rows,
  grip state line, walk status) -> ``brain.observation.WorldObservation``.
* ``gateway_node`` (rclpy Foxy, Stuart-deploy only): the ROS node that
  subscribes the skill topics, bridges the writer/grip sockets and the pick
  engine, and exposes tool verbs to the brain agent over a JSON-lines socket.

The package ``__init__`` deliberately imports nothing: ``gateway_node`` must
import cleanly on a laptop with neither rclpy nor the brain modules present,
so every heavy import is lazy inside methods.
"""

__all__ = ["assemble", "Assembler", "gateway_node"]
