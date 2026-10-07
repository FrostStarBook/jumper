"""Build and validate a jump-plus-crab-dance reference for Jumper."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from tasks.jumper.common.actuator import CORNER_SPEED
from tasks.jumper.common.constants import get_spec
from tasks.jumper.common.dance.motion import (
    LEG_JOINTS,
    SUPPORT_LEGS,
    _entity_joint_names,
    _home_support_height,
    ensure_motion_npz,
)

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "tasks/jumper/breakdance/media"
CRAB_CLIP = ROOT / "tasks/jumper/dance/media/demo.npz"
JUMP_CLIP = ROOT / "tasks/jumper/jump/ref/high_jump_flat.npz"
CONTROL_DT = 0.02
JUMP_TIME_SCALE = 1.8
DANCE_SECONDS = 6.0
BLEND_SECONDS = 0.5


def _quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.T
    return np.stack(
        (
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ),
        axis=1,
    )


def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    result = np.empty((len(rpy), 4))
    for frame, angles in enumerate(rpy):
        mujoco.mju_euler2Quat(result[frame], angles, "xyz")
    return result


def _sample_rows(values: np.ndarray, source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(target_t, source_t, values[:, column]) for column in range(values.shape[1])],
        axis=1,
    )


def _task_channel(joint_name: str) -> str:
    for leg, names in LEG_JOINTS.items():
        if joint_name in names:
            return f"L{leg}_j{names.index(joint_name)}"
    raise ValueError(f"unknown Jumper joint {joint_name!r}")


def _support_heights(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    site_ids: list[int],
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    joint_pos: np.ndarray,
) -> np.ndarray:
    heights = np.empty((len(joint_pos), len(site_ids)))
    for frame in range(len(joint_pos)):
        data.qpos[:] = 0.0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[qpos_addresses] = joint_pos[frame]
        mujoco.mj_forward(model, data)
        heights[frame] = [data.site_xpos[site, 2] for site in site_ids]
    return heights


def _build_arrays() -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    entity_joints = _entity_joint_names()
    model = get_spec(None).compile()
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in entity_joints
    ]
    qpos_addresses = np.asarray([model.jnt_qposadr[joint] for joint in joint_ids])
    site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
        for leg in SUPPORT_LEGS
    ]
    if any(site < 0 for site in site_ids):
        raise ValueError(f"support foot sites not found for {SUPPORT_LEGS}")
    home_z = _home_support_height(model, data, qpos_addresses, site_ids)

    with np.load(JUMP_CLIP, allow_pickle=True) as jump:
        metadata = json.loads(str(jump["meta"]))
        jump_order = metadata["joint_order"]
        permutation = [jump_order.index(name) for name in entity_joints]
        jump_t = np.asarray(jump["t"], dtype=np.float64)
        target_t = np.arange(
            0.0,
            jump_t[-1] * JUMP_TIME_SCALE + CONTROL_DT / 4,
            CONTROL_DT,
        )
        source_t = target_t / JUMP_TIME_SCALE
        jump_q = _sample_rows(np.asarray(jump["q"])[:, permutation], jump_t, source_t)
        jump_root_pos = _sample_rows(np.asarray(jump["base_pos"]), jump_t, source_t)
        source_rpy = _quat_to_rpy(np.asarray(jump["base_quat"]))
        jump_root_rpy = _sample_rows(
            np.unwrap(source_rpy, axis=0), jump_t, source_t
        )
        go_frame = int(np.argmin(np.abs(jump_t - float(metadata["t_go"]))))
        go_root_pos = np.asarray(jump["base_pos"])[go_frame]
        go_quat = np.asarray(jump["base_quat"])[go_frame]
        go_joint_pos = np.asarray(jump["q"])[go_frame, permutation]

    # Match the older jump model's go-pose to this checkout's standing height.
    data.qpos[:] = 0.0
    data.qpos[:3] = go_root_pos
    data.qpos[3:7] = go_quat
    data.qpos[qpos_addresses] = go_joint_pos
    mujoco.mj_forward(model, data)
    go_site_z = np.asarray(data.site_xpos)[site_ids, 2]
    jump_root_pos[:, 2] += home_z - float(np.median(go_site_z))

    crab = np.load(CRAB_CLIP)
    source_stride = round(CONTROL_DT / float(crab["dt"]))
    crab_ids = np.arange(
        0, min(len(crab["time"]), round(DANCE_SECONDS / float(crab["dt"]))), source_stride
    )
    crab_channels = [_task_channel(name) for name in entity_joints]
    crab_q = np.stack([crab[f"meas_{name}"][crab_ids] for name in crab_channels], axis=1)
    crab_root_pos = np.stack(
        [crab[f"meas_body_{axis}"][crab_ids] for axis in ("x", "y", "z")], axis=1
    )
    crab_root_rpy = np.stack(
        [crab[f"meas_body_{axis}"][crab_ids] for axis in ("roll", "pitch", "yaw")],
        axis=1,
    )
    start = int(np.argmin(np.sqrt(np.mean((crab_q - jump_q[-1]) ** 2, axis=1))))
    crab_q = crab_q[start:]
    crab_root_pos = crab_root_pos[start:]
    crab_root_rpy = crab_root_rpy[start:].copy()

    jump_root_quat = _rpy_to_quat(jump_root_rpy)
    crab_root_quat = _rpy_to_quat(crab_root_rpy)
    jump_end_site_z = _support_heights(
        model,
        data,
        qpos_addresses,
        site_ids,
        jump_root_pos[-1:],
        jump_root_quat[-1:],
        jump_q[-1:],
    )[0]
    crab_start_site_z = _support_heights(
        model,
        data,
        qpos_addresses,
        site_ids,
        crab_root_pos[:1],
        crab_root_quat[:1],
        crab_q[:1],
    )[0]

    yaw_delta = jump_root_rpy[-1, 2] - crab_root_rpy[0, 2]
    rotation = np.array(
        [[np.cos(yaw_delta), -np.sin(yaw_delta)],
         [np.sin(yaw_delta), np.cos(yaw_delta)]]
    )
    crab_root_pos[:, :2] = (
        (crab_root_pos[:, :2] - crab_root_pos[0, :2]) @ rotation.T
        + jump_root_pos[-1, :2]
    )
    crab_root_pos[:, 2] += float(np.median(jump_end_site_z) - np.median(crab_start_site_z))
    crab_root_rpy[:, 2] += yaw_delta

    dance_count = min(round(DANCE_SECONDS / CONTROL_DT), len(crab_q))
    crab_q = crab_q[:dance_count]
    crab_root_pos = crab_root_pos[:dance_count]
    crab_root_rpy = crab_root_rpy[:dance_count]
    blend_frames = round(BLEND_SECONDS / CONTROL_DT)
    blend_t = np.linspace(0.0, 1.0, blend_frames + 2)[1:-1]
    smooth = blend_t * blend_t * (3 - 2 * blend_t)
    blend_q = jump_q[-1] + smooth[:, None] * (crab_q[0] - jump_q[-1])
    blend_pos = jump_root_pos[-1] + smooth[:, None] * (
        crab_root_pos[0] - jump_root_pos[-1]
    )
    blend_rpy = jump_root_rpy[-1] + smooth[:, None] * (
        crab_root_rpy[0] - jump_root_rpy[-1]
    )
    joint_pos = np.concatenate((jump_q, blend_q, crab_q[1:]))
    root_pos = np.concatenate((jump_root_pos, blend_pos, crab_root_pos[1:]))
    root_rpy = np.concatenate((jump_root_rpy, blend_rpy, crab_root_rpy[1:]))
    root_quat = _rpy_to_quat(root_rpy)

    support_z = _support_heights(
        model,
        data,
        qpos_addresses,
        site_ids,
        root_pos,
        root_quat,
        joint_pos,
    )
    support_contact = np.abs(support_z - home_z) <= 0.015
    moving_steps = np.abs(np.diff(joint_pos, axis=0)) / CONTROL_DT
    max_speed = float(np.max(moving_steps))
    if max_speed > float(CORNER_SPEED):
        raise ValueError(
            f"reference reaches {max_speed:.2f} rad/s, above the servo corner "
            f"speed of {float(CORNER_SPEED):.2f}; increase JUMP_TIME_SCALE or "
            "BLEND_SECONDS"
        )

    source_channels = {
        "dt": np.asarray(CONTROL_DT, dtype=np.float32),
        "time": (np.arange(len(joint_pos)) * CONTROL_DT).astype(np.float32),
        "phase": np.ones(len(joint_pos), dtype=np.float32),
        "support_contact": support_contact.astype(np.float32),
    }
    for column, axis in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        value = root_pos[:, column] if column < 3 else root_rpy[:, column - 3]
        source_channels[f"body_{axis}"] = value.astype(np.float32)
        source_channels[f"meas_body_{axis}"] = value.astype(np.float32)
    for column, name in enumerate(crab_channels):
        value = joint_pos[:, column].astype(np.float32)
        source_channels[name] = value
        source_channels[f"meas_{name}"] = value

    stats: dict[str, float | int] = {
        "jump_scale": JUMP_TIME_SCALE,
        "source_dance_start_s": float(crab_ids[start] * float(crab["dt"])),
        "duration_s": float(len(joint_pos) * CONTROL_DT),
        "max_joint_speed_rad_s": max_speed,
        "corner_speed_rad_s": float(CORNER_SPEED),
        "grounded_frames": int(np.count_nonzero(support_contact.sum(axis=1) >= 3)),
        "airborne_frames": int(np.count_nonzero(support_contact.sum(axis=1) < 3)),
    }
    return source_channels, stats


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    arrays, stats = _build_arrays()
    with tempfile.TemporaryDirectory(prefix="breakdance-build-", dir=MEDIA) as temp:
        temp_dir = Path(temp)
        temp_motion = temp_dir / "breakdance.npz"
        np.savez_compressed(temp_motion, **arrays)
        ensure_motion_npz(CONTROL_DT, temp_dir)
        destination = MEDIA / temp_motion.name
        temp_motion.replace(destination)

    # Build the persistent cache only after the validated material is installed.
    ensure_motion_npz(CONTROL_DT, MEDIA)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"validated reference: {MEDIA / 'breakdance.npz'}")


if __name__ == "__main__":
    main()