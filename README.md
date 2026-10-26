<h1 align="center">OpenBot: An Embodied LLM Agent for the Unitree G1 Humanoid</h1>

<p align="center"><b>A harness for embodied language agents.</b><br>
Deterministic skills · a typed tool interface · verdicts that come from measurement, not from hope.</p>

<p align="center"><em>Final-year engineering research project · 2026</em></p>

---

Say *"hey Stuart, get me a bottle of water"*. The robot turns, walks, looks, picks, walks back and hands it
over — or it refuses and tells you why. A cloud language model decides **what** to do; a deterministic skill
layer decides **how**; and every skill reports **whether it actually worked**, measured rather than assumed.

No task-specific manipulation policy is trained for the final system. That is the point: the interesting object
here is not the policy, it is the substrate a policy could be trained against — a harness on which three learned
agents (a skill forger, a skill caller and a call reviewer) can later be post-trained. That post-training is
future work, out of reach here for compute, data and hardware-time reasons; see
[the roadmap](#what-comes-next-post-training-the-three-agents).

## The numbers

| | |
|---|---|
| Scored spoken trials | **45** across four successive releases |
| Completed, or correctly refused | **32 / 45** — Wilson 95 % CI **0.57 – 0.82** |
| Chained pick-and-place | **17 / 26** overall · **15 / 17** across the last three releases |
| Container chains on the final release | **3 / 3** completed |
| Spoken stop → motion halted | **5 – 19 ms** after the gateway receives it |
| Median first response | **~1 s** |
| False-success skill verdicts in trial review | **0** |
| Simulation grasp acquisitions | **99 / 100** with lift under ±1 cm noise |

## Why this exists

Two ways to make a humanoid manipulate, both unsatisfying.

**Learn it end-to-end.** A visuomotor policy that maps pixels to joints needs enormous demonstration data, and
when it fails it cannot tell you why. We spent a full phase of this project proving that point to ourselves.

**Script it.** A hand-written stack of motions is verifiable and safe, but it cannot be steered by a person
speaking English, and it cannot generalise past its branches.

The third position is the one this project takes: **make the robot a harness a model can drive, measure and
improve against.** Skills become typed tools with preconditions and postconditions; each returns a verdict
derived from a measurement; refusal becomes a first-class outcome with a reason; and the whole stack runs in a
MuJoCo twin with the same interfaces, so a candidate policy can be exercised without touching hardware.

## What we tried first, and what it taught us

Part 1 was the conventional bet: a **108 M-parameter conditional U-Net diffusion policy** trained on
**2,160 scripted demonstrations (1.42 M samples)**, across **9 policy variants**, **50 hyperparameter settings**
and **14 training runs**.

The demonstrations came from a collection setup that was **not the room the robot actually works in** — partly
public data, partly scripted demonstrations. We knew that, so we tried to buy back the difference the usual way:
augmentation, noise and a wider mix of sources.
It did not survive contact with the real scene. On the robot the policy did not complete the task, and across
**2,089 evaluation trials the lift rate was 0 %**.

The post-mortem named two failures that were happening at once:

- **A domain gap that augmentation cannot close.** Widening the data does not make a policy robust to a camera
  pose, a table height, a lighting condition or a set of objects it has never seen — it only makes the training
  distribution wider. When the deployment scene differs structurally from the collection scene, the policy has
  no reason to transfer, and no way to notice that it has not.
- **A proprioceptive shortcut.** The policies were reading their own joint state as a proxy for success: in
  training, that state predicted the label without the policy ever learning to use the object. A policy that can
  satisfy its loss this way looks trained and behaves blindly.

Joint-space reinforcement learning converged to a hovering optimum with **0 % contact**; `mjx` ran out of memory
on an 8 GB laptop GPU. Generating motions directly from the model was rejected on measured latency: the local 4B
model managed under 4 tokens/s, 75–85 s per decision.

The lesson is not "learning does not work". It is that a policy which cannot say *why* it failed gives you
nothing to debug, and a policy trained off-distribution cannot be trusted to generalise into a room it has never
seen. That is why the final system puts the model at the **task level**, keeps motion deterministic, and makes
every skill report a verdict that can be checked.

## The harness

Four layers, each with exactly one job.

```
                 voice  ·  "hey Stuart, get me a bottle of water"
                   │
        ┌──────────▼───────────┐
        │  1  speech-to-speech │   cloud agent: hears, answers, emits tool calls
        └──────────┬───────────┘
                   │  typed calls, JSON schema, strict vocabulary
        ┌──────────▼───────────┐
        │  2  tool gateway     │   validates, gates, routes, holds the interlocks
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │  3  deterministic    │   localise · pick · place · walk · descend · stop
        │      skills          │   each returns a measurement, never an assumption
        └──────────┬───────────┘
                   │
        ┌──────────▼───────────┐
        │  4  whole-body       │   one writer, 35 slots at 250 Hz
        │      controller      │   GR00T Balance + Walk at 50 Hz inside that loop
        └──────────┬───────────┘
                   │
             Unitree G1 "Stuart"
```

### What a verdict looks like

Success is not "the command was sent". For a grasp, the evidence is the **closure-block margin**: how far the
fingers stop short of the close preset. Measured on the robot, an **empty hand reads 0.038 rad** while a hand
**holding an object reads 0.828 – 1.256 rad**, and the verifier's tolerance is **0.06 rad**. Tactile pressure
never reached an evidence threshold on an empty hand versus thirteen objects, so it is retained only as
corroboration.

The same pattern repeats across the library:

- **Pick** solves the reach before any motor command and refuses above a **0.15 m IK residual**; it checks the
  planned path against the body and the tucked left hand, and refuses any target crossing the midline.
- **Walk** stops on a measured range and reports `REACHED` only from its own state machine.
- **Place** verifies release, and the report is explicit that release is not containment.

### Design principles

1. **The model plans; it never moves.** Every motion comes from a deterministic skill.
2. **Every skill verifies itself.** Reach, closure, grasp, arrival and release each produce a measurement.
3. **Refusal is an outcome, not an error.** Out of reach, hand full, object lost — the robot says which.
4. **Safety sits outside the model.** Interlocks and watchdogs bound motion independently; spoken stop bypasses
   the language model entirely and halted reaching **5–19 ms** after the gateway received it.
5. **Simulation before hardware.** A MuJoCo twin of the G1 — with **seven model defects corrected** — validates
   motion first.

## Engineering notes

- **Depth on the robot.** The RealSense D435i is read through a direct **V4L2 Z16** pipeline rather than the
  vendor stack, which removed a layer that could not be kept in sync with the control loop.
- **Vision at 30 Hz.** YOLO11n-seg runs as a TensorRT engine on the Orin NX; the open-vocabulary YOLOE variant
  uses a text-embedding bank with per-target filtering.
- **The 250 Hz jitter.** Arm command latency was traced to **DDS/GIL starvation** from high-rate subscribers
  inside the same interpreter; removing those listeners fixed a jitter that no amount of tuning had touched.
- **One writer.** A single process owns all body motors: 35 slots at **250 Hz**, with the balance and walking
  policy executed every fifth tick (**50 Hz**); the hand server runs separately at 20 Hz.
- **Grasp without crushing.** Radius-based stand-off (object radius plus **5 cm**), bounded adaptive closure and
  torque-aware limits keep the motors out of a latch.
- **Test discipline.** The first stack carried a **309-test** suite; the final tree holds over a hundred test files
  and more than a thousand test functions, sixteen integration scenarios with a simulated body, and mutation
  campaigns that killed all their mutants.

## Demo videos

Each recording is a 2x2 grid captured from the same run: **third person** (top left), the robot's own
**eyes** (top right), the **map** (bottom left) and the **trace** of what the planner believed (bottom right).
The line above each video is what was said to the robot.

### Fetch a bottle of water

> "Can you go get me a bottle of water?"

https://github.com/user-attachments/assets/f6764f5c-ef31-4619-b22c-f82881fda42b

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__get_me_a_bottle_of_water_3rd_POVBottle.mp4)

### Pick the closest cup, stop, pick the farthest

> "Pick the closest cup."  ·  "Stop."  ·  "Pick the farthest cup."

https://github.com/user-attachments/assets/9d4d92ff-0f09-4ffc-8f26-b57e1ecb0e62

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__pick_closestCUP_STOP_pick_farthestCUP_3rd_POV.mp4)

### Pick the cup closest to the bottle

> "Pick the cup closest to the bottle."

https://github.com/user-attachments/assets/abe8a86f-24c4-4112-847f-e3e413ce4ab5

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__pick_CUP%28OUTof3CUPs%29_closest_toBottle_3rd_POV.mp4)

### Pick the cyan cup of three

> "Pick the cyan cup."

https://github.com/user-attachments/assets/e8d2e216-bcb5-4987-ade3-3233d5963dce

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__pick_cyan%28green%29_coloured_cup%28OUTof3Cups%29_3rd_POV.mp4)

### Pick and place bottle to bottle

> "Pick up the bottle and put it next to the other bottle."

https://github.com/user-attachments/assets/754ba5f9-b774-41a8-af57-fcc6da3e6032

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__pick_place_bottle_to_bottle_3rd_POV.mp4)

### Pick red cup, stop, pick green

> "Pick up the red cup."  ·  "Stop."  ·  "Pick up the green cup."

https://github.com/user-attachments/assets/c6ec8527-4168-491c-ba7b-6781a9e89106

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__pick_redCUP_stop_pick_greenCUP_OUTof3Cups_3rd_POV.mp4)

### Pick up an apple

> "Pick up the apple."

https://github.com/user-attachments/assets/83281a3f-dde7-4074-925b-c9b7aecb45ae

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__pick_up_apple_3rd_POV.mp4)

### Transport a bottle to the bin

> "Put that bottle in the bin."

https://github.com/user-attachments/assets/37ba5d07-881f-492c-b4ca-99b39ee58db7

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__transport_bottle_to_bin_3rd_POVBin.mp4)

### Banana transport

> "Pick up the banana and find a bowl to put in."

https://github.com/user-attachments/assets/83999446-343d-4ecc-98bb-db3195b23640

[Full recording](https://github.com/enqinghayashi/OpenBot/releases/download/demos-v1/ppt-demos__transporting_banana_to_bowl_in_another_lower_workspace.mp4)

## Runtime interfaces

- **Writer socket** `/tmp/merged_writer.sock` — `cmd_set vx vy omega [dur]`, pose verbs, `arm <14f>`, `q`,
  `armstate`, `hold_rc on|off`, `freeze`, `status`, `quit`. Precedence: `hold_rc|shutdown > cmd_auto > RC`.
- **Grip socket** `/tmp/grip_server.sock` — `open|close [side]`, `close_adaptive`, `state [side]`.
- **Engine socket** `/tmp/pick_engine.sock` — `pick x y z [n] [orient] [height]`, `place` / `place_left`,
  `drop`, `open|close_both`. One engine per pick lane; the gateway attaches so a place continues from the pose
  the pick left.
- **Walk topics** `/walk_target`, `/walk_go`, `/walk_status`; `REACHED` is the only arrival signal.
- **Detections** `/object_detections_body` — event driven; silence means an empty scene.

## Configuration

`config.yaml` is the base profile; `config.voice.yaml`, `config.offline.yaml`, `config.cloud.yaml`,
`config.stuart.yaml` and `config.stuart-cuda.yaml` layer on top. On the robot, `config.yaml` carries
device-specific values and is maintained in place rather than overwritten from a development copy.

## Requirements

- Python 3.8 on the robot (`rclpy`); current Python for laptop tooling.
- CycloneDDS on robot and companion machine (configurations in `skills/config/`).
- TensorRT on the on-board Jetson for the vision models.
- MuJoCo for simulation and motion validation.
- Unitree SDK2 for the locomotion lane.

## Deploying to the robot

The stack runs on the robot's on-board computer. Choose any directory there as the deploy root; the examples use
three shell variables:

```bash
ROBOT=unitree@<robot-address>                                # ssh target
ROBOT_ROOT=/home/unitree/openbot                             # deploy root on the robot
ROBOT_PYTHON=/home/unitree/miniconda3/envs/tv/bin/python3    # interpreter the stack runs with
```

**What ships**

*Manifest push* — `config.yaml`, `config.cloud.yaml`, `config.voice.yaml`, `config.offline.yaml`, `brain/`,
`gateway/`, `voice/`.

*Skills lane* — copied by hand into the deploy root: `tabletop_pick_place_engine.py`, `pick_engine_v3.py`,
`grip_server.py`, `pick_node.py`, `walk_node.py`, `walk_lock.py`, `yolo_node.py`, `depth_projector_node.py`,
`openbot_camera_node.py`, `main_merged_writer.py`, `pick_walk_coordinator.py`, `arrival_relay.py`,
`restore_arm_pose.py`, `fsm_check.py`, the launchers from `skills/launchers/`, and `setup_openbot.sh` from
`skills/config/`.

**Step 1 — build the manifest**

```bash
python3 deploy/make_manifest.py
```

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

All three must pass: the remote md5 matches, the null-byte count is `0`, and `py_compile` succeeds with the
interpreter the stack uses. A mismatch means a corrupted transfer — copy again and re-verify before restarting.

## Running on the robot

Three terminals, started in this order.

**Terminal 1 — the stack**

```bash
source "$ROBOT_ROOT/setup_openbot.sh"
cd "$ROBOT_ROOT" && bash launch_openbot_stack.sh
```

Starts the merged writer, then the perception and pick-engine chain. Wait for both before starting the gateway.

**Terminal 2 — gateway**

```bash
exec bash --norc --noprofile
source "$ROBOT_ROOT/setup_openbot.sh"
export OPENBOT_CREEP_EDGE_M=0.35
cd "$ROBOT_ROOT" && OPENBOT_PROFILE=offline OPENBOT_PICK_RADIUS_PARK=1 \
  python3 -c "import sys; sys.argv = ['gateway_node', '--ros-args', \
'-p', 'open_vocab:=true']; from gateway.gateway_node import main; \
sys.exit(main(['--config', 'config.yaml']))" 2>&1 \
  | tee /tmp/pick_walk_logs/gateway.log
```

The gateway must stay attached: the ask/confirm path reads its stdin, and a detached gateway times out after 20 s.

**Terminal 3 — voice app**

```bash
source "$ROBOT_ROOT/setup_openbot.sh"
cd "$ROBOT_ROOT" && bash voice/run_voice.sh
```

Logs land in `/tmp/pick_walk_logs/`. To stop, send `quit` to the writer socket and stop the voice app.

## Evaluation and limits

The numbers above come from a pre-registered protocol of **45 scored spoken trials across four releases**, not
from a single frozen configuration. Read them with these bounds:

- Small samples, and the release changed during the protocol.
- Compliance with the task-specific time limits was **not** independently rescored.
- One container chain lacked a recorded landing; the place verifier establishes **release, not containment**.
- Early releases produced false failures; later ones produced no false successes in the review.
- Hand-full refusal succeeded in only one of three trials, and recorded motor temperatures exceeded the pause
  threshold.
- Several constants were set by judgement rather than optimisation: the 0.55 m walk stop gate, the 0.35 voice
  confidence floor, the 5 cm stand-off margin, the 0.06 rad closure tolerance, the 0.30 m self-zone and the 2 s
  freshness budget.

Open work: evaluation on one frozen release, containment verification, registered depth, state preservation
across cloud failover, active thermal protection, bimanual skills, odometry and gateway authentication.

## What comes next: post-training the three agents

The harness is deliberately model-agnostic. Its end state — the reason it is a harness rather than a demo — is an
agent *trained against* it, not merely driving it. Three models would close the loop.

**1. The skill-forging agent.** Given a capability gap ("the hand cannot reach a bottle lying on the floor"), it
proposes a new skill against the typed interface, runs it in the MuJoCo twin, reads the verification verdict and
iterates until the postcondition holds. The twin and the measurement-based verdict are what make the loop
self-checking: a candidate that merely *looks* right still fails its postcondition, so plausible motion cannot
pass for success.

**2. The invoking agent.** It learns to drive the harness the way a competent operator does — decomposing a
spoken request, ordering calls, retrying after a measured failure, recovering from a refusal, asking the person a
question when the request is ambiguous, and stopping when the evidence says stop. Today a general-purpose cloud
agent does this zero-shot under prompt constraints. The trained version would learn the calling convention a
prompt cannot teach: when to re-perceive, when to re-plan, when to give up.

**3. The reviewing agent.** It learns to audit the invoking agent's calls *before* they reach the gateway —
rejecting calls whose preconditions cannot hold, whose geometry is infeasible, or whose ordering leaves the robot
in a bad state. This moves verification from a post-hoc verdict to a pre-flight check, and is the closest analogue
to an experienced operator reading a plan and saying "no, that will collide".

Trained together they form a closed improvement loop: the forger extends the library, the invoker learns to use
it, the reviewer keeps both honest against the body's constraints.

**Why it is not in this repository.** Reaching that point needs post-training compute and data a final-year
project did not have: rollouts at scale on the twin, a reward or preference signal built on the verification
verdicts, and enough hardware time to validate candidate skills on the robot. The harness was built first,
deliberately, because it is the part that makes a training signal trustworthy. The three-agent loop is the next
project, not a missing feature.

## Background

Final-year engineering research project, 2026. Supervisor: Professor Thomas Bräunl.
