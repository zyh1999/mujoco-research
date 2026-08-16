"""Grouped subprocess VecEnv for large MuJoCo batches.

Unlike Stable-Baselines3's SubprocVecEnv, this implementation runs several
environments serially inside each worker.  That keeps a 2,048-environment
logical vector while using a bounded number of CPU processes.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
from typing import Any, Callable, Iterable, Optional, Sequence

import cloudpickle
import numpy as np


class _CloudpickleWrapper:
    def __init__(self, value: Any):
        self.value = value

    def __getstate__(self) -> bytes:
        return cloudpickle.dumps(self.value)

    def __setstate__(self, state: bytes) -> None:
        self.value = pickle.loads(state)


def _reset_one(env: Any, seed: Optional[int]) -> tuple[Any, dict[str, Any]]:
    result = env.reset(seed=seed) if seed is not None else env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _step_one(env: Any, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        info = dict(info)
        info["TimeLimit.truncated"] = bool(truncated and not terminated)
    elif len(result) == 4:
        observation, reward, done, info = result
        done = bool(done)
        info = dict(info)
    else:
        raise ValueError(f"Unexpected env.step() result length: {len(result)}")

    if done:
        info["terminal_observation"] = observation
        observation, reset_info = _reset_one(env, seed=None)
        if reset_info:
            info["reset_info"] = reset_info
    return observation, float(reward), done, info


def _worker(
    remote: Any,
    parent_remote: Any,
    wrapped_env_fns: _CloudpickleWrapper,
) -> None:
    parent_remote.close()
    envs = [env_fn() for env_fn in wrapped_env_fns.value]
    try:
        while True:
            command, payload = remote.recv()
            if command == "get_spaces":
                env = envs[0]
                remote.send((env.observation_space, env.action_space, getattr(env, "spec", None)))
            elif command == "reset":
                seeds = payload
                remote.send([_reset_one(env, seed)[0] for env, seed in zip(envs, seeds)])
            elif command == "step":
                remote.send([_step_one(env, action) for env, action in zip(envs, payload)])
            elif command == "close":
                remote.close()
                break
            else:
                raise NotImplementedError(command)
    except EOFError:
        pass
    finally:
        for env in envs:
            env.close()


class GroupedSubprocVecEnv:
    """A minimal VecEnv-compatible grouped subprocess backend.

    ``num_workers`` controls OS process count; logical ``num_envs`` remains the
    length of ``env_fns``.  Workers receive contiguous slices, preserving the
    original environment ordering and seed-to-rank mapping.
    """

    def __init__(
        self,
        env_fns: Iterable[Callable[[], Any]],
        num_workers: int = 11,
        start_method: str = "forkserver",
    ) -> None:
        env_fns = list(env_fns)
        if not env_fns:
            raise ValueError("GroupedSubprocVecEnv requires at least one environment")
        if not 1 <= num_workers <= len(env_fns):
            raise ValueError(
                f"num_workers must be in [1, {len(env_fns)}], got {num_workers}"
            )

        self.num_envs = len(env_fns)
        self.num_workers = int(num_workers)
        self.closed = False
        self.waiting = False
        self.viewer = None
        self._pending_seeds: list[Optional[int]] = [None] * self.num_envs

        index_groups = [group.tolist() for group in np.array_split(np.arange(self.num_envs), self.num_workers)]
        self._slices = [np.asarray(group, dtype=np.int64) for group in index_groups]
        env_groups = [[env_fns[index] for index in group] for group in index_groups]

        context = mp.get_context(start_method)
        self.remotes, worker_remotes = zip(*[context.Pipe() for _ in env_groups])
        self.processes = [
            context.Process(
                target=_worker,
                args=(worker_remote, remote, _CloudpickleWrapper(env_group)),
                daemon=True,
            )
            for remote, worker_remote, env_group in zip(self.remotes, worker_remotes, env_groups)
        ]
        for process in self.processes:
            process.start()
        for worker_remote in worker_remotes:
            worker_remote.close()

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space, self.spec = self.remotes[0].recv()

    def seed(self, seed: Optional[int] = None) -> Sequence[Optional[int]]:
        if seed is None:
            max_base_seed = int(np.iinfo(np.uint32).max) - self.num_envs
            seed = int(np.random.randint(0, max_base_seed, dtype=np.uint32))
        self._pending_seeds = [int(seed) + rank for rank in range(self.num_envs)]
        return self._pending_seeds

    def reset(self) -> np.ndarray:
        self._assert_open()
        for remote, indices in zip(self.remotes, self._slices):
            remote.send(("reset", [self._pending_seeds[index] for index in indices]))
        observations = [remote.recv() for remote in self.remotes]
        self._pending_seeds = [None] * self.num_envs
        return _stack_observations(_flatten(observations))

    def step_async(self, actions: np.ndarray) -> None:
        self._assert_open()
        if self.waiting:
            raise RuntimeError("step_async() called while another step is pending")
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")
        for remote, indices in zip(self.remotes, self._slices):
            remote.send(("step", actions[indices]))
        self.waiting = True

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self._assert_open()
        if not self.waiting:
            raise RuntimeError("step_wait() called without step_async()")
        grouped_results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        results = _flatten(grouped_results)
        observations, rewards, dones, infos = zip(*results)
        return (
            _stack_observations(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.bool_),
            list(infos),
        )

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
            self.waiting = False
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.closed = True

    def _assert_open(self) -> None:
        if self.closed:
            raise RuntimeError("GroupedSubprocVecEnv is closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _flatten(groups: Iterable[Iterable[Any]]) -> list[Any]:
    return [item for group in groups for item in group]


def _stack_observations(observations: Iterable[Any]) -> Any:
    observations = list(observations)
    if isinstance(observations[0], dict):
        return {
            key: np.stack([observation[key] for observation in observations])
            for key in observations[0]
        }
    return np.stack(observations)
