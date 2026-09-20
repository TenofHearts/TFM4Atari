"""Gymnasium Atari construction compatible with RL Zoo DQN checkpoints."""

from __future__ import annotations

from collections import deque
from contextlib import redirect_stderr
from io import StringIO
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tfm4atari.config import AtariConfig, RuntimeConfig


class ChannelFrameStack(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Stack grayscale frames along their channel axis for SB3 policies."""

    def __init__(self, env: gym.Env[np.ndarray, int], n_stack: int = 4) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("Atari observation space must be a Box")
        shape = env.observation_space.shape
        if shape is None or len(shape) != 3 or shape[-1] != 1:
            raise ValueError(f"Expected grayscale HWC observation, got {shape}")
        low = np.repeat(env.observation_space.low, n_stack, axis=-1)
        high = np.repeat(env.observation_space.high, n_stack, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
        self._frames: deque[np.ndarray] = deque(maxlen=n_stack)
        self._n_stack = n_stack

    def _stack(self) -> np.ndarray:
        return np.concatenate(tuple(self._frames), axis=-1)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._frames.clear()
        self._frames.extend(observation.copy() for _ in range(self._n_stack))
        return self._stack(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(observation)
        return self._stack(), float(reward), terminated, truncated, info


def build_env(
    env_id: str,
    atari: AtariConfig,
    runtime: RuntimeConfig,
    *,
    render: bool = False,
) -> gym.Env[np.ndarray, int]:
    """Construct the deterministic legacy-style protocol used by RL Zoo DQN."""
    # Importing SB3 with the legacy ``gym`` compatibility package installed emits
    # an unconditional deprecation notice. Keep CLI JSON clean while retaining
    # that package solely for old checkpoint deserialization.
    with redirect_stderr(StringIO()):
        from stable_baselines3.common.atari_wrappers import AtariWrapper

    gym.register_envs(ale_py)
    render_mode: str | None = None
    if render or runtime.record_video:
        render_mode = "rgb_array"
    elif runtime.render_mode != "none":
        render_mode = runtime.render_mode
    base = gym.make(
        env_id,
        obs_type="rgb",
        frameskip=atari.frameskip,
        repeat_action_probability=atari.repeat_action_probability,
        full_action_space=atari.full_action_space,
        render_mode=render_mode,
    )
    processed = AtariWrapper(
        base,
        noop_max=atari.noop_max,
        frame_skip=atari.wrapper_frame_skip,
        screen_size=atari.screen_size,
        terminal_on_life_loss=atari.terminal_on_life_loss,
        clip_reward=False,
    )
    return ChannelFrameStack(processed, n_stack=4)


def current_ram(env: gym.Env[Any, Any]) -> np.ndarray:
    """Return a defensive copy of the console's 128-byte RAM."""
    return np.asarray(env.unwrapped.ale.getRAM(), dtype=np.uint8).copy()


def current_lives(env: gym.Env[Any, Any]) -> int:
    return int(env.unwrapped.ale.lives())


def action_meanings(env: gym.Env[Any, Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in env.unwrapped.get_action_meanings())
