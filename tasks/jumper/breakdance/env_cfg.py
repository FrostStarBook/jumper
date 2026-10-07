"""Environment config for the generated jump-and-crab-dance reference."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..dance.env_cfg import env_cfg as dance_env_cfg

MEDIA = Path(__file__).resolve().parent / "media"
EPISODE_S = 12.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Build this task using the calibrated dance-tracking baseline."""
    return dance_env_cfg(
        asset=asset,
        play=play,
        media=MEDIA,
        episode_s=EPISODE_S,
    )