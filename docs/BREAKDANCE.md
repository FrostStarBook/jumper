# Local Breakdance Training

This task trains an original simulation choreography: a slowed high jump followed
by crab-dance footwork. The linked video is a style reference only; no video-to-joint
motion capture is claimed. Training and replay run locally in MuJoCo.

Run these commands from the repository root. The existing macOS setup uses `.venv`
with Python 3.12 and the native CPU backend.

## Build And Check The Reference

Build the reference once, or after changing either source clip:

```bash
.venv/bin/python tools/build_breakdance_reference.py
```

The builder writes `tasks/jumper/breakdance/media/breakdance.npz` only after joint
limits, servo speed, contact geometry, and motion conversion checks pass. Confirm
that the task is registered and its configuration resolves:

```bash
.venv/bin/python scripts/train.py --list
.venv/bin/python scripts/train.py --task jumper.breakdance --backend native --device cpu --num_envs 64 --dry-run
```

## Preview The Reference

On macOS, `mjpython` is required for the live MuJoCo viewer. This plays the reference
without a trained policy, which is the quickest way to see the complete routine:

```bash
.venv/bin/mjpython scripts/play.py --task jumper.breakdance --backend native --device cpu --num_envs 1 --viewer-env-num 1 --agent zero
```

This is fixed choreography playback, not joystick teleoperation. Use the MuJoCo
viewer mouse controls to orbit, pan, and zoom the camera.

## Train

Start a full run with 64 parallel CPU environments and no viewer:

```bash
.venv/bin/python scripts/train.py --task jumper.breakdance --backend native --device cpu --num_envs 64 --headless --no-tensorboard
```

The task is configured for 10,000 PPO iterations. A small smoke run can check the
training path without waiting for a useful policy:

```bash
.venv/bin/python scripts/train.py --task jumper.breakdance --backend native --device cpu --num_envs 64 --headless --no-tensorboard --max-iterations 3
```

Each run writes checkpoints under `logs/jumper/jumper.breakdance/<timestamp>/`.
Three iterations only prove that the environment and optimizer run; they are not a
trained dance policy.

## Resume

Continue from the newest checkpoint for another 1,000 iterations:

```bash
.venv/bin/python scripts/train.py --task jumper.breakdance --backend native --device cpu --num_envs 64 --headless --no-tensorboard --resume --max-iterations 1000
```

Or choose a specific checkpoint:

```bash
.venv/bin/python scripts/train.py --task jumper.breakdance --backend native --device cpu --num_envs 64 --headless --no-tensorboard --checkpoint logs/jumper/jumper.breakdance/<timestamp>/model_999.pt
```

`--checkpoint` resumes automatically. The run directory contains the checkpoints
and TensorBoard event files when logging is enabled.

## Replay A Trained Policy

With no checkpoint argument, replay selects the newest checkpoint for this task:

```bash
.venv/bin/mjpython scripts/play.py --task jumper.breakdance --backend native --device cpu --num_envs 1 --viewer-env-num 1
```

To replay a particular checkpoint, add `--checkpoint <path>/model_<iteration>.pt`.
The policy is being judged against the reference; quality improves with training and
is not implied by a successful smoke run.