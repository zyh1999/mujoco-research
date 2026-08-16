#!/usr/bin/env python3
"""Record matched HumanoidStandup checkpoint rollouts.

The script is intentionally tied to the original trust-region MLP checkpoint
format: 256x2 actor/critic, observation RMS, PopArt critic, and vector logstd.
It records both policies from the same reset seed and saves per-step actions so
the video comparison has an accompanying numerical trace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import importlib.util
import numpy as np
import torch

try:
    import mujoco
except ModuleNotFoundError:
    mujoco = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--high-checkpoint", type=Path, required=True)
    parser.add_argument("--ordinary-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--backend", choices=("direct", "gymnasium"), default="direct")
    return parser.parse_args()


def checkpoint_state(path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")
    return payload


def make_model(
    repo: Path,
    observation_dim: int,
    action_dim: int,
    checkpoint: Path,
    device: torch.device,
):
    sys.path.insert(0, str(repo))
    from utils.utils import ActorCritic, build_mlp  # pylint: disable=import-error

    nets_config = types.SimpleNamespace(
        norm_obs=True,
        a_dropout=0.0,
        c_dropout=0.0,
        a_hidden_size=256,
        a_num_layers=2,
        c_hidden_size=256,
        c_num_layers=2,
    )
    observation_space = types.SimpleNamespace(shape=(observation_dim,))
    fn_neural_nets, _ = build_mlp(observation_space, device=device)
    model = ActorCritic(
        fn_neural_nets,
        observation_space.shape,
        nets_config=nets_config,
        n_actions=None,
        dim_actions=action_dim,
        with_popart=True,
        sigma_type="vector",
        device=device,
    ).to(device)
    model.load_state_dict(checkpoint_state(checkpoint, device), strict=True)
    model.eval()
    return model


def policy_action(
    model,
    obs: np.ndarray,
    device: torch.device,
    action_mode: str,
    noise: np.ndarray | None,
) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        if model.obs_rms is not None:
            obs_t = model.obs_rms(obs_t)
        outputs = model.forward_pi(obs_t)
        mean, logstd = outputs.chunk(2, dim=-1)
        if action_mode == "sample":
            if noise is None:
                raise ValueError("Sampled actions require a noise vector")
            noise_t = torch.as_tensor(noise[None], dtype=mean.dtype, device=device)
            action = mean + torch.exp(logstd) * noise_t
        else:
            action = mean
    return action.squeeze(0).cpu().numpy()


class HumanoidStandupReplay:
    """Minimal Gymnasium-v4-compatible dynamics/observation adapter.

    Using MuJoCo's supported Python binding keeps CPU rendering independent of
    the legacy mujoco_py extension (which requires unavailable OSMesa headers
    on CSF3's CPU nodes). The observation concatenation matches
    HumanoidStandup-v4: qpos[2:], qvel, cinert, cvel, actuator forces, and
    external contact forces.
    """

    def __init__(self, xml_path: Path, seed: int, width: int = 640, height: int = 480):
        if mujoco is None:
            raise RuntimeError("The direct backend requires the native mujoco Python package")
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.initial_qpos = self.model.qpos0.copy()
        self.initial_qvel = np.zeros(self.model.nv, dtype=np.float64)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.camera.distance = 3.6
        self.camera.azimuth = 90.0
        self.camera.elevation = -12.0
        self.reset(seed)

    @property
    def observation_dim(self) -> int:
        return self.model.nq - 2 + self.model.nv + self.model.nbody * 10 + self.model.nbody * 6 + self.model.nv + self.model.nbody * 6

    @property
    def action_dim(self) -> int:
        return self.model.nu

    def observation(self) -> np.ndarray:
        return np.concatenate(
            (
                self.data.qpos.flat[2:],
                self.data.qvel.flat,
                self.data.cinert.flat,
                self.data.cvel.flat,
                self.data.qfrc_actuator.flat,
                self.data.cfrc_ext.flat,
            )
        ).astype(np.float32, copy=False)

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos + rng.uniform(-0.01, 0.01, size=self.model.nq)
        self.data.qvel[:] = self.initial_qvel + rng.uniform(-0.01, 0.01, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        action = np.clip(action, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = action
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        # Gymnasium's v4 implementation scores absolute torso height each
        # control step, rather than the temporal height difference.
        uph_cost = float(self.data.qpos[2]) / self.model.opt.timestep
        ctrl_cost = 0.1 * float(np.square(action).sum())
        impact_cost = min(0.5e-6 * float(np.square(self.data.cfrc_ext).sum()), 10.0)
        return self.observation(), uph_cost + 1.0 - ctrl_cost - impact_cost

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self) -> None:
        # MuJoCo 2.3's Python Renderer does not expose an explicit close().
        return None


class GymnasiumHumanoidStandupReplay:
    """Adapter around the exact Gymnasium/MuJoCo stack used by the trainer."""

    def __init__(self, seed: int, render: bool):
        import gymnasium as gym

        kwargs = {"render_mode": "rgb_array"} if render else {}
        self.env = gym.make("HumanoidStandup-v4", **kwargs)
        self.ctrl_low = self.env.action_space.low.copy()
        self.ctrl_high = self.env.action_space.high.copy()
        self.reset(seed)

    @property
    def observation_dim(self) -> int:
        return int(self.env.observation_space.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.env.action_space.shape[0])

    @property
    def data(self):
        return self.env.unwrapped.data

    def reset(self, seed: int) -> np.ndarray:
        observation, _ = self.env.reset(seed=seed)
        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float]:
        observation, reward, _, _, _ = self.env.step(action.astype(np.float32, copy=False))
        return observation, float(reward)

    def render(self) -> np.ndarray:
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("Gymnasium environment did not return an RGB frame")
        return frame

    def close(self) -> None:
        self.env.close()


class VideoWriter:
    """Write RGB frames through the bundled ffmpeg binary, without imageio's plugin API."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        import imageio_ffmpeg

        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def append_data(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        assert self.process.stdin is not None
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None
        self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited {return_code}: {stderr[-1000:]}")


def summary(
    actions: list[np.ndarray], rewards: list[float], returns: list[float], action_limit: np.ndarray
) -> dict:
    action_array = np.asarray(actions, dtype=np.float32)
    return {
        "steps": int(len(rewards)),
        "reward_sum": float(np.sum(rewards)),
        "episode_returns": [float(value) for value in returns],
        "action_rms": float(np.sqrt(np.mean(np.square(action_array)))),
        "action_abs_mean": float(np.mean(np.abs(action_array))),
        "action_abs_max": float(np.max(np.abs(action_array))),
        "action_saturation_fraction": float(np.mean(np.abs(action_array) >= 0.99 * np.abs(action_limit))),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.backend == "gymnasium":
        high_env = GymnasiumHumanoidStandupReplay(args.seed, render=not args.no_video)
        ordinary_env = GymnasiumHumanoidStandupReplay(args.seed, render=not args.no_video)
    else:
        gymnasium_spec = importlib.util.find_spec("gymnasium")
        if gymnasium_spec is None or gymnasium_spec.origin is None:
            raise RuntimeError("gymnasium is required to locate humanoidstandup.xml")
        xml_path = Path(gymnasium_spec.origin).parent / "envs" / "mujoco" / "assets" / "humanoidstandup.xml"
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)
        high_env = HumanoidStandupReplay(xml_path, args.seed)
        ordinary_env = HumanoidStandupReplay(xml_path, args.seed)
    if high_env.observation_dim != 376 or high_env.action_dim != 17:
        raise RuntimeError(f"Unexpected HumanoidStandup shape: obs={high_env.observation_dim}, act={high_env.action_dim}")
    high_model = make_model(args.repo, high_env.observation_dim, high_env.action_dim, args.high_checkpoint, device)
    ordinary_model = make_model(args.repo, ordinary_env.observation_dim, ordinary_env.action_dim, args.ordinary_checkpoint, device)

    high_actions: list[np.ndarray] = []
    ordinary_actions: list[np.ndarray] = []
    high_rewards: list[float] = []
    ordinary_rewards: list[float] = []
    high_heights: list[float] = []
    ordinary_heights: list[float] = []
    high_returns: list[float] = []
    ordinary_returns: list[float] = []
    rng = np.random.default_rng(args.action_seed)
    high_writer = ordinary_writer = paired_writer = None
    if not args.no_video:
        height, width = high_env.render().shape[:2]
        high_writer = VideoWriter(args.output_dir / "high.mp4", width, height, args.fps)
        ordinary_writer = VideoWriter(args.output_dir / "ordinary.mp4", width, height, args.fps)
        paired_writer = VideoWriter(args.output_dir / "high_vs_ordinary.mp4", width * 2, height, args.fps)
    try:
        for episode in range(args.episodes):
            high_obs = high_env.reset(args.seed + episode)
            ordinary_obs = ordinary_env.reset(args.seed + episode)
            high_return = 0.0
            ordinary_return = 0.0
            for _ in range(args.steps):
                noise = rng.standard_normal(high_env.action_dim) if args.action_mode == "sample" else None
                # Match utils.runners.Runner exactly: sample in unconstrained
                # Normal coordinates, then squash into the MuJoCo action range.
                high_action = np.tanh(
                    policy_action(high_model, high_obs, device, args.action_mode, noise)
                ) * high_env.ctrl_high
                ordinary_action = np.tanh(
                    policy_action(ordinary_model, ordinary_obs, device, args.action_mode, noise)
                ) * ordinary_env.ctrl_high
                high_obs, high_reward = high_env.step(high_action)
                ordinary_obs, ordinary_reward = ordinary_env.step(ordinary_action)
                if high_writer is not None and ordinary_writer is not None and paired_writer is not None:
                    high_frame = high_env.render()
                    ordinary_frame = ordinary_env.render()
                    high_writer.append_data(high_frame)
                    ordinary_writer.append_data(ordinary_frame)
                    paired_writer.append_data(np.concatenate([high_frame, ordinary_frame], axis=1))

                high_actions.append(high_action)
                ordinary_actions.append(ordinary_action)
                high_rewards.append(high_reward)
                ordinary_rewards.append(ordinary_reward)
                high_heights.append(float(high_env.data.qpos[2]))
                ordinary_heights.append(float(ordinary_env.data.qpos[2]))
                high_return += high_reward
                ordinary_return += ordinary_reward
            high_returns.append(high_return)
            ordinary_returns.append(ordinary_return)
    finally:
        if high_writer is not None:
            high_writer.close()
        if ordinary_writer is not None:
            ordinary_writer.close()
        if paired_writer is not None:
            paired_writer.close()
        high_env.close()
        ordinary_env.close()

    np.savez_compressed(
        args.output_dir / "action_traces.npz",
        high_actions=np.asarray(high_actions),
        ordinary_actions=np.asarray(ordinary_actions),
        high_rewards=np.asarray(high_rewards),
        ordinary_rewards=np.asarray(ordinary_rewards),
        high_heights=np.asarray(high_heights),
        ordinary_heights=np.asarray(ordinary_heights),
    )
    high_report = summary(high_actions, high_rewards, high_returns, high_env.ctrl_high)
    ordinary_report = summary(ordinary_actions, ordinary_rewards, ordinary_returns, ordinary_env.ctrl_high)
    high_height_array = np.asarray(high_heights).reshape(args.episodes, args.steps)
    ordinary_height_array = np.asarray(ordinary_heights).reshape(args.episodes, args.steps)
    high_report["episode_max_heights"] = [float(value) for value in high_height_array.max(axis=1)]
    high_report["episode_final_heights"] = [float(value) for value in high_height_array[:, -1]]
    ordinary_report["episode_max_heights"] = [float(value) for value in ordinary_height_array.max(axis=1)]
    ordinary_report["episode_final_heights"] = [float(value) for value in ordinary_height_array[:, -1]]
    report = {
        "environment": f"HumanoidStandup-v4 {args.backend} replay",
        "seed": args.seed,
        "episodes": args.episodes,
        "action_mode": args.action_mode,
        "action_seed": args.action_seed if args.action_mode == "sample" else None,
        "high_checkpoint": str(args.high_checkpoint.resolve()),
        "ordinary_checkpoint": str(args.ordinary_checkpoint.resolve()),
        "high": high_report,
        "ordinary": ordinary_report,
        "paired_video_layout": "high policy left; ordinary policy right",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
