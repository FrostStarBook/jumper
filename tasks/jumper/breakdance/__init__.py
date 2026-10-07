"""jumper.breakdance -- imitate a jump followed by crab-dance footwork."""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.breakdance",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating a validated jump-and-dance reference; "
    "build it with tools/build_breakdance_reference.py",
    tags=("imitation", "jumper", "dance", "jump", "flat"),
)