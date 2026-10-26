# OpenBot

Control stack for the **Unitree G1** humanoid robot. A local, tool-calling LLM **brain** directs a deterministic
**skill lane** through a **gateway**: voice in; walking, pick-and-place, perception and whole-body motion out.
The model selects tools and arguments — motion itself is produced entirely by the deterministic lane: no learned
action policy in the control loop, and no cloud services.

## Architecture

```
        voice / text input
                │
        ┌───────▼────────┐
        │     brain      │   LLM tool-calling loop, tool contract,
        │                │   verifier, scene graph
        └───────┬────────┘
                │  tool calls (JSON)
        ┌───────▼────────┐
        │    gateway     │   schema / strict-vocabulary checks,
        │                │   safety gates, motion routing
        └───────┬────────┘
                │
     ┌──────────┼───────────────┬────────────────┐
     ▼          ▼               ▼                ▼
 ┌────────┐ ┌──────────┐ ┌────────────┐ ┌────────────────┐
 │ writer │ │   arm    │ │ locomotion │ │   perception   │
 │whole-  │ │ grip +   │ │ walk FSM,  │ │ YOLO/SEG +     │
 │body    │ │ pick &   │ │ hunt law,  │ │ TensorRT,      │
 │motion  │ │ place    │ │ odometry   │ │ depth projector│
 └────────┘ └──────────┘ └────────────┘ └────────────────┘
                Unitree G1 hardware
```

## Repository layout

```
OpenBot/
├── brain/              harness layer: agent loop, tool contract, verifier, scene graph, LLM clients
├── gateway/            brain <-> skills bridge (rclpy, Python 3.8), descent/container logic
├── voice/              voice app (realtime + cascade), mic/speaker, keyword spotting, echo gate
├── skills/             the deterministic skill lane
│   ├── writer/         whole-body merged writer (35 slots, 250 Hz), GR00T policies, MuJoCo model + meshes
│   ├── arm/            grip server, pick/place engines, grasp geometry, hand-state lane
│   ├── perception/     YOLO/SEG + TensorRT nodes, depth projector, pointcloud, operator viewers
│   ├── locomotion/     walk-node FSM, hunt law, command daemon
│   ├── odom/           odometry + detection gate
│   ├── coordination/   pick–walk coordinator, arrival relay/announce
│   ├── sonic/          SONIC motion core
│   ├── writer_box/     pose-verb builder
│   ├── helpers/        pre-start checks, pose utilities
│   ├── launchers/      stack launch scripts
│   ├── config/         CycloneDDS configurations, environment setup
│   └── probes/         on-robot calibration tools
├── robot/              byte-exact copies of the files that run on the deployed robot
├── deploy/             manifest tool and model bake-off harness
└── config*.yaml        runtime profiles (see Configuration)
```

## Runtime interfaces

- **Writer socket** `/tmp/merged_writer.sock` — `cmd_set vx vy omega [dur]` (walk), pose verbs, `arm <14f>`, `q`,
  `armstate`, `hold_rc on|off`, `freeze`, `status`, `quit`.
  Precedence: `hold_rc|shutdown > cmd_auto > RC`.
- **Grip socket** `/tmp/grip_server.sock` — `open|close [side]`, `close_adaptive`, `state [side]`
  (verification signal; the grip server does not auto-verify).
- **Engine socket** `/tmp/pick_engine.sock` — `pick x y z [n] [orient] [height]`, `place` / `place_left`,
  `drop` (release-only), `open|close_both`. One engine instance per pick lane; the gateway attaches so a place
  continues from the pose the pick left. Pick targets are tabletop objects.
- **Walk topics** — `/walk_target`, `/walk_go`, `/walk_status`; `REACHED` is the only verification signal.
- **Detections** `/object_detections_body` — event-driven; silence means an empty scene.

## Configuration

- `config*.yaml` — runtime profiles: `config.yaml` (default), `config.voice.yaml`, `config.offline.yaml`,
  `config.cloud.yaml`, `config.stuart.yaml`, `config.stuart-cuda.yaml`.
- On the robot, `config.yaml` carries device-specific values and is maintained in place rather than overwritten
  from the development copy.

## Requirements

- Python 3.8 on the robot runtime (`rclpy`); current Python for laptop-side tooling.
- CycloneDDS on the robot and the companion machine (configurations in `skills/config/`).
- TensorRT on the on-board Jetson for the vision models.
- MuJoCo for the simulation and verification tooling.
- Unitree SDK2 for the locomotion lane.

## Deploying to the robot

The stack runs on the robot's on-board computer. Choose any directory there as the deploy root; the examples
below use three shell variables:

```bash
ROBOT=unitree@<robot-address>                                # ssh target
ROBOT_ROOT=/home/unitree/openbot                             # deploy root on the robot
ROBOT_PYTHON=/home/unitree/miniconda3/envs/tv/bin/python3    # interpreter the stack runs with
```

**What ships**

*Manifest push* — `config.yaml`, `config.cloud.yaml`, `config.voice.yaml`, `config.offline.yaml`, `brain/`,
`gateway/`, `voice/`.

*Skills lane* — copied by hand into the deploy root, because it is not part of the manifest:
`tabletop_pick_place_engine.py`, `pick_engine_v3.py`, `grip_server.py`, `pick_node.py`, `walk_node.py`,
`walk_lock.py`, `yolo_node.py`, `depth_projector_node.py`, `openbot_camera_node.py`, `main_merged_writer.py`,
`pick_walk_coordinator.py`, `arrival_relay.py`, `restore_arm_pose.py`, `fsm_check.py`, plus the launchers from
`skills/launchers/` and `setup_openbot.sh` from `skills/config/`.

**Step 1 — build the manifest**

```bash
python3 deploy/make_manifest.py
python3 -c "import json;d=json.load(open('deploy/manifest.json'));print(len(d['files']),'files')"
```

The manifest records each file's local md5 and its target path, and is the input to the copy step.

**Step 2 — copy each file**

```bash
FILE=<local path>
REL=<path under $ROBOT_ROOT>

LAPTOP_MD5=$(md5sum "$FILE" | awk '{print $1}')
scp "$FILE" "$ROBOT:$ROBOT_ROOT/$REL"
```

**Step 3 — verify on the robot, every file**

```bash
timeout 15 sshpass -p "$ROBOT_PASS" ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=8 "$ROBOT" \
  "md5sum $ROBOT_ROOT/$REL | awk '{print \$1}'; \
   tr -cd '\\0' < $ROBOT_ROOT/$REL | wc -c; \
   $ROBOT_PYTHON -m py_compile $ROBOT_ROOT/$REL"
```

All three must pass: the remote md5 equals `$LAPTOP_MD5`, the null-byte count is `0`, and `py_compile` succeeds
with the interpreter the stack actually uses. Any mismatch means a corrupted transfer — copy again and re-verify
before restarting anything.

## Running on the robot

Start from a clean shell so the robot's DDS configuration is picked up rather than a desktop one:

```bash
ssh "$ROBOT"
exec bash --norc --noprofile
source "$ROBOT_ROOT/setup_openbot.sh"
cd "$ROBOT_ROOT"
```

Then start the stack in this order:

```bash
bash launch_openbot_stack.sh     # 1. whole-body writer + pick/walk chain (foreground)
bash launch_pick_stack.sh        # 2. perception, grip server, camera
python3 -m gateway.gateway_node  # 3. gateway (brain <-> skills bridge)
bash voice/run_voice.sh          # 4. voice app
```

The writer must be up before the gateway: the gateway attaches to the writer socket and fails its first command
otherwise. Logs land in `/tmp/pick_walk_logs/` (`writer.log`, `grip.log`, `camera.log`).

To stop, send `quit` to the writer socket (`/tmp/merged_writer.sock`) and stop the voice app; the launchers exit
when their foreground process exits.
