"""Build a procedural Thomas-flare-style reference for Jumper.

Morphology mapping (not human joint copy):
  - Front arms LF/RF: reach forward/down (plant / press feel)
  - Root: lowered, continuous yaw spin
  - Mid/rear legs LM/RM/LR/RR: phased circular footwork near the ground
    (required so ensure_motion_npz support checks can pass)

Writes tasks/jumper/breakdance/media/breakdance.npz after validation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mujoco
import numpy as np

from tasks.jumper.common.actuator import CORNER_SPEED
from tasks.jumper.common.constants import HOME, STAND_Z, get_spec
from tasks.jumper.common.dance.motion import (
    LEG_JOINTS,
    SUPPORT_LEGS,
    _entity_joint_names,
    _home_support_height,
    ensure_motion_npz,
)

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "tasks/jumper/breakdance/media"
OUT_NAME = "breakdance.npz"

CONTROL_DT = 0.02
DURATION_S = 10.0
NUM_SPINS = 2.0  # full yaw revolutions over the clip
BODY_DROP_M = 0.02  # how much lower than STAND_Z
FRONT_REACH = 0.5  # rad-scale blend of front-arm plant pose
LEG_SWING = 0.1  # mid/rear joint swing amplitude (rad)
CIRCLE_RADIUS_XY = 0.5  # small root orbit (m); 0 = pure spin in place
PITCH_RAD = 0.002  # slight nose-down

# Soft ease-in / ease-out of the spin amplitude at the ends (seconds)
EASE_S = 0.6


def _task_channel(joint_name: str) -> str:
    for leg, names in LEG_JOINTS.items():
        if joint_name in names:
            return f"L{leg}_j{names.index(joint_name)}"
    raise ValueError(f"unknown Jumper joint {joint_name!r}")


def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    result = np.empty((len(rpy), 4))
    for frame, angles in enumerate(rpy):
        mujoco.mju_euler2Quat(result[frame], angles, "xyz")
    return result


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _spin_envelope(t: np.ndarray) -> np.ndarray:
    """1 in the middle, ramps 0→1 and 1→0 at the ends."""
    env = np.ones_like(t)
    if EASE_S <= 0:
        return env
    fade_in = _smoothstep01(t / EASE_S)
    fade_out = _smoothstep01((DURATION_S - t) / EASE_S)
    return fade_in * fade_out


def _home_vector(entity_joints: tuple[str, ...]) -> np.ndarray:
    return np.asarray([HOME[name] for name in entity_joints], dtype=np.float64)


def _front_plant_offsets() -> dict[str, float]:
    """Extra joint offsets (added to HOME) that reach the front arms forward/down.

    Signs follow the V1.6 HOME / limit layout (left/right mirrored).
    Tuned to stay inside soft limits; reduce FRONT_REACH if conversion fails.
    """
    return {
        # Left front
        "LF_J0_joint": -0.25,
        "LF_J1_joint": -0.35,
        "LF_J2_joint": -0.20,
        "LF_J3_joint": -0.15,
        "LF_J4_joint": 0.0,
        # Right front (mirrored)
        "RF_J0_joint": 0.25,
        "RF_J1_joint": -0.35,
        "RF_J2_joint": 0.20,
        "RF_J3_joint": 0.15,
        "RF_J4_joint": 0.0,
    }


def _build_arrays() -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    entity_joints = _entity_joint_names()
    n_joints = len(entity_joints)
    model = get_spec(None).compile()
    data = mujoco.MjData(model)

    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in entity_joints
    ]
    if any(j < 0 for j in joint_ids):
        missing = [n for n, j in zip(entity_joints, joint_ids) if j < 0]
        raise ValueError(f"joints not in model: {missing}")
    qpos_addresses = np.asarray([model.jnt_qposadr[j] for j in joint_ids])

    support_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg) for leg in SUPPORT_LEGS
    ]
    if any(s < 0 for s in support_site_ids):
        raise ValueError(f"support foot sites not found for {SUPPORT_LEGS}")

    front_site_ids = {
        leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
        for leg in ("LF", "RF")
    }
    if any(s < 0 for s in front_site_ids.values()):
        raise ValueError("front foot sites LF/RF not found")

    home_z = _home_support_height(model, data, qpos_addresses, support_site_ids)
    home_q = _home_vector(entity_joints)
    plant = _front_plant_offsets()

    n = int(round(DURATION_S / CONTROL_DT))
    t = np.arange(n, dtype=np.float64) * CONTROL_DT
    env = _spin_envelope(t)
    phase = 2.0 * np.pi * NUM_SPINS * (t / DURATION_S)

    # --- root trajectory ---
    yaw = phase * env
    root_pos = np.zeros((n, 3), dtype=np.float64)
    root_pos[:, 0] = CIRCLE_RADIUS_XY * np.cos(yaw)
    root_pos[:, 1] = CIRCLE_RADIUS_XY * np.sin(yaw)
    root_pos[:, 2] = STAND_Z - BODY_DROP_M

    root_rpy = np.zeros((n, 3), dtype=np.float64)
    root_rpy[:, 1] = PITCH_RAD * env  # pitch
    root_rpy[:, 2] = yaw

    # --- joint trajectory from HOME + plant + phased leg swings ---
    joint_pos = np.tile(home_q, (n, 1))

    # Front arms: blend toward plant pose with a slow press pulse
    press = 0.5 + 0.5 * np.sin(phase)  # 0..1
    for name, delta in plant.items():
        col = entity_joints.index(name)
        joint_pos[:, col] = home_q[col] + FRONT_REACH * press * env * delta

    # 中后腿：只动 J0（几乎不改变足高），J1/J2 锁 HOME，方便过共面检查
    swing_legs = {
        "LM": 0.0,
        "RM": 0.5 * np.pi,
        "LR": np.pi,
        "RR": 1.5 * np.pi,
    }
    for leg, phi0 in swing_legs.items():
        col0 = entity_joints.index(f"{leg}_J0_joint")
        col1 = entity_joints.index(f"{leg}_J1_joint")
        col2 = entity_joints.index(f"{leg}_J2_joint")
        phi = phase + phi0
        joint_pos[:, col0] = home_q[col0] + LEG_SWING * env * np.sin(phi)
        joint_pos[:, col1] = home_q[col1]   # 不动
        joint_pos[:, col2] = home_q[col2]   # 留给共面修正 / 保持 HOME

        # --- 把四条 SUPPORT 腿的脚尖拉回共面（改 J2）---
    support_leg_j2 = {
        "LM": entity_joints.index("LM_J2_joint"),
        "RM": entity_joints.index("RM_J2_joint"),
        "LR": entity_joints.index("LR_J2_joint"),
        "RR": entity_joints.index("RR_J2_joint"),
    }
    # J2 增大/减小与足端高度的近似方向（左右腿符号相反）
    j2_sign = {"LM": -1.0, "RM": 1.0, "LR": -1.0, "RR": 1.0}
    site_by_leg = {
        leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, leg)
        for leg in SUPPORT_LEGS
    }

    for frame in range(n):
        for _ in range(6):  # 少量迭代即可
            data.qpos[:] = 0.0
            data.qpos[:3] = root_pos[frame]
            data.qpos[3:7] = _rpy_to_quat(root_rpy[frame : frame + 1])[0]
            data.qpos[qpos_addresses] = joint_pos[frame]
            mujoco.mj_forward(model, data)
            zs = np.array([data.site_xpos[site_by_leg[leg], 2] for leg in SUPPORT_LEGS])
            target = float(np.median(zs))
            if float(np.max(zs) - np.min(zs)) < 0.003:
                break
            for leg in SUPPORT_LEGS:
                err = data.site_xpos[site_by_leg[leg], 2] - target
                col = support_leg_j2[leg]
                # 约 1 rad ≈ 把脚抬/放几厘米，按误差比例收
                joint_pos[frame, col] += j2_sign[leg] * (-4.0 * err)
                jid = joint_ids[col]
                lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
                joint_pos[frame, col] = np.clip(
                    joint_pos[frame, col], lo + 0.02, hi - 0.02
                )

    # Clamp to model joint limits (soft margin)
    for col, jid in enumerate(joint_ids):
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        margin = 0.02
        joint_pos[:, col] = np.clip(joint_pos[:, col], lo + margin, hi - margin)

    root_quat = _rpy_to_quat(root_rpy)

    # --- lift/drop root so median SUPPORT foot site stays near home_z ---
    for frame in range(n):
        data.qpos[:] = 0.0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[qpos_addresses] = joint_pos[frame]
        mujoco.mj_forward(model, data)
        support_z = np.array([data.site_xpos[s, 2] for s in support_site_ids])
        root_pos[frame, 2] += home_z - float(np.median(support_z))

    root_quat = _rpy_to_quat(root_rpy)

    # Final FK for contact masks and stats
    support_z = np.empty((n, len(support_site_ids)))
    front_z = np.empty((n, 2))
    for frame in range(n):
        data.qpos[:] = 0.0
        data.qpos[:3] = root_pos[frame]
        data.qpos[3:7] = root_quat[frame]
        data.qpos[qpos_addresses] = joint_pos[frame]
        mujoco.mj_forward(model, data)
        support_z[frame] = [data.site_xpos[s, 2] for s in support_site_ids]
        front_z[frame, 0] = data.site_xpos[front_site_ids["LF"], 2]
        front_z[frame, 1] = data.site_xpos[front_site_ids["RF"], 2]

    support_contact = (np.abs(support_z - home_z) <= 0.015).astype(np.float64)

    speeds = np.abs(np.diff(joint_pos, axis=0)) / CONTROL_DT
    max_speed = float(np.max(speeds)) if len(speeds) else 0.0
    if max_speed > float(CORNER_SPEED):
        raise ValueError(
            f"reference reaches {max_speed:.2f} rad/s, above corner "
            f"{float(CORNER_SPEED):.2f}; reduce NUM_SPINS, LEG_SWING, or FRONT_REACH"
        )

    crab_channels = [_task_channel(name) for name in entity_joints]
    source_channels: dict[str, np.ndarray] = {
        "dt": np.asarray(CONTROL_DT, dtype=np.float32),
        "time": t.astype(np.float32),
        "phase": np.ones(n, dtype=np.float32),
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
        "duration_s": float(n * CONTROL_DT),
        "num_spins": float(NUM_SPINS),
        "body_drop_m": float(BODY_DROP_M),
        "max_joint_speed_rad_s": max_speed,
        "corner_speed_rad_s": float(CORNER_SPEED),
        "grounded_support_frames": int(
            np.count_nonzero(support_contact.sum(axis=1) >= 3)
        ),
        "mean_front_site_z": float(np.mean(front_z)),
        "mean_support_site_z": float(np.mean(support_z)),
        "home_support_z": float(home_z),
    }
    return source_channels, stats


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    arrays, stats = _build_arrays()
    with tempfile.TemporaryDirectory(prefix="thomas-flare-build-", dir=MEDIA) as temp:
        temp_dir = Path(temp)
        temp_motion = temp_dir / OUT_NAME
        np.savez_compressed(temp_motion, **arrays)
        ensure_motion_npz(CONTROL_DT, temp_dir)
        destination = MEDIA / OUT_NAME
        temp_motion.replace(destination)

    ensure_motion_npz(CONTROL_DT, MEDIA)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"validated reference: {MEDIA / OUT_NAME}")


if __name__ == "__main__":
    main()
