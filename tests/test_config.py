from pathlib import Path

import pytest

from tfm4atari.config import ProjectConfig, load_config


def test_default_config_enables_only_beamrider() -> None:
    config = load_config(Path(__file__).parents[1] / "config.toml")
    assert [game.name for game in config.active_games] == ["BeamRider"]
    assert config.active_games[0].teacher_backend == "qrdqn"
    assert len(config.games) == 7
    assert config.context_cache.teacher_judgment_capacity == 16
    assert config.context_cache.judgment_capacity == 8
    assert config.context_cache.refit_every_judged_batches == 32
    assert config.video.max_decisions == 1024
    assert config.runtime.cpu_context_rows == 4000
    assert config.runtime.cuda_context_rows == 10000


def test_unknown_enabled_game_is_rejected() -> None:
    config = load_config(Path(__file__).parents[1] / "config.toml")
    raw = config.model_dump()
    raw["enabled_games"] = ("NotAGame",)
    with pytest.raises(ValueError, match="unknown games"):
        ProjectConfig.model_validate(raw)
