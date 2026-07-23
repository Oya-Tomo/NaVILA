# NaVILA Zenoh Node

[日本語](README_ja.md)

This directory provides two independent processes:

- `node.py` receives camera JPEGs, maintains NaVILA's observation history, runs inference, and safely translates the current result into Go2 commands.
- `cli.py` sends an instruction, `start`, `stop`, and a heartbeat to the NaVILA node. It never accesses Go2 keys directly.

The camera and robot bridges are separate projects and separate processes:

- [camera-zenoh-node](https://github.com/Oya-Tomo/camera-zenoh-node)
- [unitree-go2-zenoh-node](https://github.com/Oya-Tomo/unitree-go2-zenoh-node)

They are not submodules or vendored dependencies. This repository does not clone, install, start, stop, or supervise either project. Prepare each project in its own checkout by following its documentation.

> [!WARNING]
> This software commands physical hardware. First test with recorded JPEGs and no robot. For physical tests, support the Go2, use low configured speeds, keep the remote control available, and establish an emergency-stop procedure. Running another publisher on `unitree/go2/command` at the same time is unsupported and is not technically prevented.

## Data flow and ownership

```text
camera-zenoh-node               NaVILA node                     unitree-go2-zenoh-node
camera/front (JPEG) ----------> frame history
                                inference/current action ------> unitree/go2/command
                         +----> lifecycle/state <--------------- unitree/go2/state
                         |
operator <----> CLI -----+
             navila/command
             navila/state
```

The NaVILA node owns model inference and the high-level `stand`, velocity, zero, and `down` requests. The Go2 node owns the observed-state posture workflow, including `StopMove`, quiescence checks, and `StandDown`. The CLI owns operator interaction only.

There is no strict distributed startup order: the NaVILA node remains idle if the camera, Go2 node, or CLI is absent. The recommended order is camera and Go2 nodes, then the NaVILA node, then the CLI, because readiness is visible immediately.

## Setup

From the NaVILA repository root:

```console
$ uv sync --group zenoh
$ cp zenoh/node-config.example.json5 zenoh/node-config.json5
$ cp zenoh/zenoh-config.example.json5 zenoh/zenoh-config.json5
$ cp zenoh/cli-config.example.json5 zenoh/cli-config.json5
$ cp zenoh/cli-zenoh-config.example.json5 zenoh/cli-zenoh-config.json5
```

Edit all four runtime files. `node-config.json5` and `cli-config.json5` are strictly validated by Pydantic. The two `*-zenoh-config.json5` files are passed directly to Zenoh and configure mode, listen/connect endpoints, and discovery.

These four local runtime files are ignored by Git to avoid committing host-specific endpoints or settings; the `*.example.json5` templates remain tracked.

Settings must agree across the independent projects:

| Purpose | NaVILA setting | External setting | Example key |
| --- | --- | --- | --- |
| Camera JPEG input | `camera.key` | camera `base_key/device_key` | `camera/front` |
| NaVILA control/state | both `zenoh_key_prefix` values | CLI only | `navila/command`, `navila/state` |
| Go2 commands/state | `go2.zenoh_key_prefix` | Go2 `zenoh_key_prefix` | `unitree/go2/command`, `unitree/go2/state` |
| Network transport | both NaVILA Zenoh files | both external Zenoh files | matching peers/router/discovery |

All keys must be concrete Zenoh keys; wildcards are rejected. The CLI example is a client connecting to the NaVILA node at `127.0.0.1:7447`; replace that address when the CLI is remote. If multiple peer processes run on one host, do not configure them all to listen on the same TCP port. Use distinct listeners, multicast discovery, or a shared router as appropriate for the deployment.

The NaVILA node consumes the current Go2 `develop` State fields `robot_connected`, `robot_state.state`, and `accepting_commands`. Legacy State payloads using `connected` and `robot` are rejected.

The programs intentionally expose only configuration-file arguments. Model, keys, rates, and motion values cannot be overridden individually on the command line.

## Run

Start the inference node from the repository root:

```console
$ uv run --group zenoh python zenoh/node.py \
    --node-config zenoh/node-config.json5 \
    --zenoh-config zenoh/zenoh-config.json5
```

Then start the independent operator CLI:

```console
$ uv run --group zenoh python zenoh/cli.py \
    --cli-config zenoh/cli-config.json5 \
    --zenoh-config zenoh/cli-zenoh-config.json5
```

The CLI sends a heartbeat every 0.2 seconds by default. The NaVILA node accepts `instruction` and `start` only after a heartbeat is fresh, but always accepts `stop`.

Example session:

```text
navila> ins Go to the elevator
Instruction: Go to the elevator
navila> status
lifecycle=idle connected=True ready=True instruction='Go to the elevator'
navila> start
Start requested; waiting for Go2 ready_stand.
Navigation started.
Running - press Enter to stop
Stopped; Go2 is down.
navila> ins Turn toward the open doorway
navila> start
```

- `ins <text>` changes the instruction while idle or in a recoverable error. `ins` by itself clears it.
- `start` asks the NaVILA node to stand, wait for `ready_stand`, and begin inference.
- `status` prints the latest NaVILA state. State is reported disconnected after `node_state_timeout_seconds` without an update.
- Press Enter at the running prompt to stop, wait for down confirmation, and return to the CLI prompt.
- Press `Ctrl+C` to exit. If the state is not confirmed idle, the CLI first sends `stop` and waits up to `command_timeout_seconds` (15 seconds in the example). It reports an unconfirmed stop but exits after the deadline.

The instruction starts as an empty string, so `start` initially remains unavailable. It is retained across stop/start cycles and is never persisted to disk.

## Timing and image history

Camera publishing, frame sampling, inference, and velocity publishing are separate rates:

| Operation | Example rate | Behavior |
| --- | ---: | --- |
| Camera callback | external camera rate, often 20–30 Hz | Validates incoming `image/jpeg`; does no inference |
| History append (`camera.sample_frequency_hz`) | 1 Hz | The callback appends only after one full sampling period since the last successful append |
| Inference (`inference.frequency_hz`) | 1 Hz | Runs only in `running`, on an absolute monotonic schedule |
| Velocity publish (`go2_velocity_publish_frequency_hz`) | 20 Hz | Repeats the current action, or zero before/after an action |
| State publish (`node_state_publish_frequency_hz`) | 20 Hz | Publishes periodically even when nothing changes |

Thus, camera frames are not taken by a separate exactly-on-the-clock sampler. Under a regular camera stream they are approximately evenly spaced, but the rule is a minimum interval from the last successful append. Missed intervals are not replayed. This keeps callbacks cheap and prevents a catch-up backlog.

The history deque size comes from `model.config.num_video_frames`; it is not duplicated in configuration. An 8-frame or 64-frame checkpoint therefore selects its own history length. One fresh real image is enough to start. A short history is left-padded with black images, and the latest real image remains the current observation. History continues to update while idle and survives instruction changes and stop/start cycles. Previous actions and model outputs are not added to the prompt.

An inference overrun skips missed schedule slots, and inference calls never overlap. A stop received during inference immediately changes the lifecycle out of `running`; when that inference returns, its result is discarded. A new start is not accepted until the stopped inference is complete and Go2 is confirmed down.

## Lifecycle and safety behavior

The normal lifecycle is:

```text
initializing -> idle -> standing -> running -> stopping -> idle
                                                 \-> error
```

Start requires a fresh CLI heartbeat, a non-empty instruction, a fresh valid camera image, a fresh connected Go2 state that accepts commands, and confirmed Go2 `down`. The node sends `stand` once and does not infer or publish velocity until Go2 reports `ready_stand` and accepts commands.

While running, the latest parsed action replaces the previous action. Supported outputs are `stop`, `move forward`, `turn left`, and `turn right`. Distances are bounded to 25/50/75 cm and turns to 15/30/45 degrees. Backward, lateral, malformed, or ambiguous output never falls back to forward motion.

All stop causes use one path: operator Enter, CLI heartbeat loss, model `stop`, stale camera, stale/disconnected Go2 state, parse/inference failure, or NaVILA `SIGINT`/`SIGTERM`. The node immediately leaves `running`, clears the current action, sends one zero-velocity command, and—only when Go2 state is fresh—sends one reliable `down` request. It then waits for both Go2 `down` and any in-flight inference.

If Go2 state is absent for `go2.node_state_timeout_seconds`, the node stops sending posture guesses and enters `error`. If down is not confirmed before `go2.down_timeout_seconds`, it remains non-moving and reports seating as unconfirmed. A recovered `error` can accept `start` only after every start condition is valid again, including confirmed `down`. Node shutdown uses the same stop path and exits nonzero if safe completion cannot be confirmed.

At startup, the first fresh Go2 state triggers one `down` request if the robot is not already down, even when no CLI is connected. The node publishes no Go2 velocity while idle.

## Validation

Validation is intentionally split by architecture. On an x86_64 development host, the test suite validates configuration and wire models, the controller state machine, action parsing, frame sampling/padding, scheduling calculations, CLI acknowledgements, and mocked Zenoh QoS/lifecycle behavior. These tests use a fake inference engine and do **not** validate the Jetson CUDA runtime or real NaVILA inference. The lock file keeps separate platform selections: ordinary PyPI Torch on non-aarch64 hosts and the configured CUDA 13.2 index on aarch64.

Run the Zenoh tests from the repository root:

```console
$ uv run --group zenoh pytest zenoh/tests
```

On the target Jetson, separately verify `uv sync --group zenoh`, import the aarch64 CUDA Torch/Torchvision and Zenoh wheels, load the configured 4bit/8bit/fp16 checkpoint, and run inference with the intended frame count while monitoring GPU memory and inference latency. An x86_64 pass is not evidence that these Jetson-specific checks pass.

Before a physical test, run a Jetson smoke test with a recorded-JPEG publisher and no robot. Then validate stand, low-speed forward/turn, Enter stop, heartbeat-loss stop, and down while the Go2 is supported and an emergency stop is immediately available.
