import pandas as pd

from tfm4atari.config import load_config
from tfm4atari.storage import DataStore


def test_incomplete_episode_is_not_resumable(tmp_path) -> None:
    config = load_config()
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"data_dir": tmp_path})}
    )
    store = DataStore(config)
    path = store.episode_path("BeamRider", "incomplete")
    path.parent.mkdir(parents=True)
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    assert store.episode_ids("BeamRider") == ()


def test_only_complete_cold_start_episodes_are_selected(tmp_path) -> None:
    config = load_config()
    config = config.model_copy(
        update={"paths": config.paths.model_copy(update={"data_dir": tmp_path})}
    )
    store = DataStore(config)
    rows = pd.DataFrame({"value": [1]})
    store.write_episode(
        "BeamRider",
        "cold",
        rows,
        {
            "collection_role": "cold_start_teacher",
            "complete_episode": True,
            "teacher": {"backend": "qrdqn"},
        },
    )
    store.write_episode(
        "BeamRider",
        "smoke",
        rows,
        {"collection_role": "smoke_test", "complete_episode": False},
    )
    assert store.cold_start_episode_ids("BeamRider", "qrdqn") == ("cold",)
    assert store.cold_start_episode_ids("BeamRider", "dqn") == ()
