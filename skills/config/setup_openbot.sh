#!/bin/bash
# setup_openbot.sh -- the ONE environment script for every OpenBot terminal on the robot.
# Box path: /home/unitree/leon/setup_openbot.sh     Use:  source /home/unitree/leon/setup_openbot.sh
#
# ROS 2 Foxy + CycloneDDS + the Unitree SDK2 workspace, CycloneDDS as the RMW with the box's
# cyclonedds.xml, ROS 2's own Python first on PYTHONPATH.  Every OpenBot node runs as a plain
# file from /home/unitree/leon (the camera too: openbot_camera_node.py), so no other ROS
# workspace is sourced.
unset ROS_DISTRO
unset AMENT_PREFIX_PATH
unset ROS_PACKAGE_PATH
unset PYTHONPATH
unset CMAKE_PREFIX_PATH

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

source /opt/ros/foxy/setup.bash
source /home/unitree/cyclonedds_ws/install/setup.bash
source /home/unitree/unitree_sdk2_ws/install/setup.bash

# ROS 2's Python first - must come after all sourcing
export PYTHONPATH=/opt/ros/foxy/lib/python3.8/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/foxy/lib:$LD_LIBRARY_PATH
export CYCLONEDDS_URI=file:///home/unitree/cyclonedds.xml
