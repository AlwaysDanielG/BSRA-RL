"""Evaluate the random baseline and a saved policy on the same reset seeds."""

import argparse
from collections.abc import Iterable, Sequence
import csv
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
import torch

from onboarding.env import make_env
from onboarding.train import runtime_versions, sha256, task_identity, validate_run_name

EVALUATION_SEEDS = tuple(range(10_000, 10_020))
ACTION_SEED_OFFSET = 20_000
VIDEO_SEED = EVALUATION_SEEDS[0]


def load_policy(path: str | Path) -> PPO:
    """Load on CPU without resuming training."""
    return PPO.load(path, device="cpu")


def choose_action(env: gym.Env, observation: NDArray[np.float64], policy: PPO | None) -> NDArray[np.float32]:
    if policy is None:
        return env.action_space.sample()
    action, _ = policy.predict(observation, deterministic=True)
    return action


def run_episode(env: gym.Env, seed: int, policy: PPO | None = None, video_path: Path | None = None, trace: bool = False) -> dict[str, Any]:
    """Supplied episode loop: reset after either ending flag, and stream video."""
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed + ACTION_SEED_OFFSET)
    total_return, length = 0.0, 0
    writer = None
    try:
        if video_path is not None:
            import imageio_ffmpeg
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio_ffmpeg.write_frames(
                str(video_path), (640, 480), fps=env.metadata["render_fps"],
                codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                ffmpeg_log_level="error",
            )
            writer.send(None)
        while True:
            action = choose_action(env, observation, policy)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            length += 1
            total_return += float(reward)
            if trace:
                print(f"seed={seed} step={length} obs={observation.tolist()} "
                      f"action={action.tolist()} reward={reward} "
                      f"terminated={terminated} truncated={truncated}")
            observation = next_observation
            if writer is not None:
                writer.send(np.ascontiguousarray(env.render()))
            if terminated or truncated:
                return dict(seed=seed, action_seed=seed + ACTION_SEED_OFFSET if policy is None else None,
                            **{"return": total_return}, length=length,
                            terminated=bool(terminated), truncated=bool(truncated))
    finally:
        if writer is not None:
            writer.close()


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, int | float]:
    """Population standard deviations describe these episodes (ddof=0)."""
    returns = np.array([row["return"] for row in rows])
    lengths = np.array([row["length"] for row in rows])
    return {
        "episodes": len(rows),
        "return_mean": float(returns.mean()), "return_std": float(returns.std()),
        "length_mean": float(lengths.mean()), "length_std": float(lengths.std()),
        "time_limit_fraction": float(np.mean([row["truncated"] for row in rows])),
    }


def evaluate(run_name: str, policy_path: str | Path | None = None, seeds: Iterable[int] = EVALUATION_SEEDS, video: bool = False, output_dir: str | Path = Path("."), trace: bool = False) -> dict[str, Any]:
    validate_run_name(run_name)
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Supply a nonempty list of distinct nonnegative reset seeds.")
    if video and VIDEO_SEED not in seeds:
        raise ValueError(f"Recording requires the predetermined seed {VIDEO_SEED}.")
    output_dir = Path(output_dir)
    results = output_dir / "results" / run_name
    torch.set_num_threads(1)
    identity = task_identity()
    policy = None
    if policy_path is not None:
        metadata = json.loads((results / "run.json").read_text())
        if metadata["task"] != identity or metadata["policy_sha256"] != sha256(policy_path):
            raise ValueError("Policy or task differs from this run's training record.")
        policy = load_policy(policy_path)
    elif (results / "run.json").exists():
        raise ValueError("Use a separate run name for random-only debugging.")
    rows = []
    summaries = {}
    video_settings = None
    labels = [("random", None)] if policy is None else [("random", None), ("ppo", policy)]
    for label, agent in labels:
        group = []
        for seed in seeds:
            record = video and seed == VIDEO_SEED and label == labels[-1][0]
            video_path = output_dir / "videos" / run_name / f"episode-{seed}.mp4" if record else None
            env = make_env(render_mode="rgb_array" if record else None)
            try:
                row = {"policy": label, **run_episode(env, seed, agent, video_path, trace)}
                group.append(row)
                if record:
                    video_settings = dict(path=str(video_path), seed=seed, policy=label,
                                          fps=env.metadata["render_fps"], width=640, height=480,
                                          frames=row["length"], codec="h264")
            finally:
                env.close()
        rows.extend(group)
        summaries[label] = summarize(group)
    summary = {
        "run_name": run_name, "seeds": list(seeds), "deterministic_policy": True,
        "action_seed_offset": ACTION_SEED_OFFSET, "std_ddof": 0,
        "task": identity, "versions": runtime_versions(),
        "policy_sha256": sha256(policy_path) if policy_path is not None else None,
        "results": summaries, "video": video_settings,
    }
    results.mkdir(parents=True, exist_ok=True)
    with (results / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy", type=Path)
    source.add_argument("--random-only", action="store_true")
    parser.add_argument("--episodes", type=int, default=20, choices=range(1, 21), metavar="1..20")
    parser.add_argument("--video", action="store_true", help="record seed 10000 within evaluation")
    parser.add_argument("--trace", action="store_true", help="print observation/action/reward/ending per step")
    args = parser.parse_args()
    summary = evaluate(args.run_name, args.policy, EVALUATION_SEEDS[:args.episodes], args.video, trace=args.trace)
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
