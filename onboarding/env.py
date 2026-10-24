"""One task definition shared by training, evaluation, and tests."""

from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "assets" / "scene.xml"
MAX_EPISODE_STEPS = 1_000
FRAME_SKIP = 2
RESET_NOISE = 0.01


def make_env(render_mode: str | None = None, max_episode_steps: int = MAX_EPISODE_STEPS) -> gym.Env:
    """Use the custom scene with Gymnasium's supplied balancing task."""
    return gym.make(
        "InvertedPendulum-v5",
        xml_file=str(SCENE),
        frame_skip=FRAME_SKIP,
        reset_noise_scale=RESET_NOISE,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        width=640,
        height=480,
        camera_name="side",
    )
