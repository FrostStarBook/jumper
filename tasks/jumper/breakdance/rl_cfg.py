"""PPO settings for the asymmetric jump-and-dance imitation task."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """Start with the existing dance-tracking settings and a separate log path."""
    return jumper_ppo_baseline(
        experiment_name="jumper.breakdance",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=10_000,
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
    )


def runner_cls() -> type:
    """Use the motion-tracking runner, which logs imitation metrics."""
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    return MotionTrackingOnPolicyRunner