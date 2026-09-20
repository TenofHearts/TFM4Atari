from pathlib import Path

from tfm4atari.config import load_config
from tfm4atari.environment import action_meanings, build_env, current_ram


def test_beamrider_environment_contract() -> None:
    config = load_config(Path(__file__).parents[1] / "config.toml")
    game = config.active_games[0]
    env = build_env(game.env_id, config.atari, config.runtime)
    observation, _ = env.reset(seed=config.runtime.seed)
    try:
        assert observation.shape == (84, 84, 4)
        assert current_ram(env).shape == (128,)
        assert len(action_meanings(env)) == env.action_space.n
    finally:
        env.close()
