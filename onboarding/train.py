"""Train one CPU PPO policy. Output/metadata helpers are supplied."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch

from onboarding.env import FRAME_SKIP, MAX_EPISODE_STEPS, RESET_NOISE, ROOT, make_env

PPO_SETTINGS = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None,
    normalize_advantage=True, ent_coef=0.0, vf_coef=0.5,
    max_grad_norm=0.5, use_sde=False, sde_sample_freq=-1,
    target_kl=None, stats_window_size=100,
)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_identity() -> dict[str, Any]:
    return {
        "environment": "InvertedPendulum-v5", "frame_skip": FRAME_SKIP,
        "max_episode_steps": MAX_EPISODE_STEPS, "reset_noise_scale": RESET_NOISE,
        "files_sha256": {name: sha256(ROOT / name) for name in (
            "assets/cartpole.xml", "assets/scene.xml", "onboarding/env.py",
        )},
    }


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{
        name: version(name) for name in (
            "gymnasium", "mujoco", "stable-baselines3", "torch", "numpy",
            "tensorboard", "imageio-ffmpeg",
        )
    }}


def validate_run_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("Use a run name containing only letters, digits, '-' and '_'.")


def train(run_name: str, seed: int = 0, total_timesteps: int = 100_000, smoke: bool = False, output_dir: str | Path = Path(".")) -> Path:
    validate_run_name(run_name)
    if total_timesteps <= 0 or seed < 0:
        raise ValueError("Steps must be positive and seed must be nonnegative.")
    output_dir = Path(output_dir)
    results = output_dir / "results" / run_name
    policy_path = output_dir / "models" / run_name / "policy.zip"
    log_dir = output_dir / "runs" / run_name
    if results.exists() or policy_path.parent.exists() or log_dir.exists():
        raise FileExistsError("Run name already used; choose a new --run-name.")
    settings = PPO_SETTINGS.copy()
    if smoke:
        total_timesteps = 256
        settings.update(n_steps=128, batch_size=64, n_epochs=2)
    torch.set_num_threads(1)
    env = Monitor(make_env())
    start = time.perf_counter()
    try:
        model = PPO(
            "MlpPolicy", env, device="cpu", seed=seed,
            tensorboard_log=str(log_dir), verbose=0, **settings,
        )

        model.learn(total_timesteps=total_timesteps, tb_log_name="ppo")

        policy_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(policy_path)

        elapsed = time.perf_counter() - start
        results.mkdir(parents=True, exist_ok=True)
        revision = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
        )
        metadata = {
            "run_name": run_name, "seed": seed, "smoke": smoke,
            "requested_steps": total_timesteps, "actual_steps": model.num_timesteps,
            "elapsed_seconds": elapsed, "device": "cpu", "torch_threads": 1,
            "policy": "MlpPolicy", "ppo": settings,
            "task": task_identity(), "versions": runtime_versions(),
            "lockfile_sha256": sha256(ROOT / "uv.lock"),
            "source_revision": revision.stdout.strip() or None,
            "source_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "training_code_sha256": sha256(Path(__file__)),
            "policy_sha256": sha256(policy_path),
            "outputs": {"policy": str(policy_path), "tensorboard": str(log_dir)},
        }
        (results / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return policy_path
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--smoke", action="store_true", help="256 steps; checks plumbing, not learning")
    args = parser.parse_args()
    steps = 256 if args.smoke else args.steps
    print(f"Training {args.run_name}: seed={args.seed}, requested steps={steps}, device=cpu", flush=True)
    print(train(args.run_name, args.seed, args.steps, args.smoke))


if __name__ == "__main__":
    main()
