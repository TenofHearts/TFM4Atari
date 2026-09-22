import pandas as pd
import pytest

from tfm4atari.config import BeamRiderCacheConfig, load_config
from tfm4atari.context_cache import ContextQueue
from tfm4atari.features import OUTCOME_CATEGORY_TARGET, OUTCOME_SCORE_TARGET
from tfm4atari.games.beamrider import (
    BeamRiderRelevanceFilter,
    label_beamrider_interval,
)
from tfm4atari.pipeline import (
    _episode_progress_feature,
    _label_symbolic_interval,
    _label_teacher_symbolic_rolling,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "online_trajectory_id": ["run", "run", "run", "run"],
            "step": [0, 1, 2, 3],
            "executed_action": [0, 1, 2, 3],
            "observed_reward": [0.0, 0.0, 44.0, 0.0],
            "lives_before": [3, 3, 3, 3],
            "lives_after": [3, 3, 3, 2],
            "symbolic_label": [0, 1, 1, -1],
            "desired_symbolic_label": [0, 1, 1, -1],
            "rolling_outcome_category": [0, 1, 1, -1],
            "rolling_outcome_score": [0.0, 0.5, 1.0, -1.0],
            "label_source": ["judge"] * 4,
            "judged_window_id": ["window"] * 4,
        }
    )


def test_beamrider_filter_keeps_every_executed_action() -> None:
    rule = BeamRiderRelevanceFilter(BeamRiderCacheConfig())
    relevant = rule.filter(_rows())
    assert relevant["step"].tolist() == [0, 1, 2, 3]

    queue = ContextQueue(maximum_rows=10)
    queue.add(relevant)
    assert queue.rows["step"].tolist() == [0, 1, 2, 3]


def test_teacher_symbolic_rolling_keeps_every_action() -> None:
    config = load_config()
    game = config.active_games[0]
    labeled = _label_teacher_symbolic_rolling(config, game, _rows())
    assert labeled["step"].tolist() == [0, 1, 2, 3]
    assert len(labeled) == len(_rows())


def test_teacher_and_online_windows_match_by_default() -> None:
    config = load_config()
    assert (
        config.context_cache.teacher_judgment_capacity
        == config.context_cache.judgment_capacity
    )


def test_episode_progress_feature_is_independent_of_episode_cap() -> None:
    config = load_config()
    short_cap = config.model_copy(
        update={
            "collection": config.collection.model_copy(
                update={"max_decisions_per_episode": 2048}
            ),
            "video": config.video.model_copy(update={"max_decisions": 1024}),
        }
    )
    assert _episode_progress_feature(config, 1024) == _episode_progress_feature(
        short_cap, 1024
    )
    assert _episode_progress_feature(short_cap, 1024) == pytest.approx(1024 / 27000)


def test_teacher_uses_overlapping_forward_credit_windows() -> None:
    config = load_config()
    config = config.model_copy(
        update={
            "context_cache": config.context_cache.model_copy(
                update={"teacher_judgment_capacity": 2, "judgment_capacity": 4}
            )
        }
    )
    rows = _rows()
    rows["observed_reward"] = [0.0, 0.0, 44.0, 0.0]
    rows["lives_after"] = rows["lives_before"]
    labeled = _label_teacher_symbolic_rolling(config, config.active_games[0], rows)
    assert labeled["symbolic_label"].tolist() == [0, 1, 1, 0]
    assert labeled[OUTCOME_CATEGORY_TARGET].tolist() == [0, 1, 1, 0]
    assert labeled[OUTCOME_SCORE_TARGET].between(-1.0, 1.0).all()
    assert labeled[OUTCOME_SCORE_TARGET].iloc[1] > 0.0
    assert labeled[OUTCOME_SCORE_TARGET].iloc[2] > 0.0
    assert labeled["rolling_window_end_step"].tolist() == [1, 2, 3, 3]
    assert labeled["rolling_window_size"].tolist() == [2, 2, 2, 1]


def test_rolling_labels_exactly_reuse_interval_label_semantics() -> None:
    config = load_config()
    game = config.active_games[0]
    rows = pd.concat([_rows()] * 5, ignore_index=True)
    rows["step"] = range(len(rows))
    rolling = _label_teacher_symbolic_rolling(config, game, rows)
    capacity = config.context_cache.teacher_judgment_capacity
    expected = [
        int(
            _label_symbolic_interval(
                config, game, rows.iloc[start : start + capacity]
            )["symbolic_label"].iloc[0]
        )
        for start in range(len(rows))
    ]
    assert rolling["symbolic_label"].tolist() == expected


def test_context_queue_is_fifo_and_deduplicates_executed_actions() -> None:
    queue = ContextQueue(maximum_rows=2)
    assert queue.add(_rows().iloc[:2]) == 2
    assert queue.add(_rows().iloc[1:]) == 2
    assert queue.rows["step"].tolist() == [2, 3]
    assert queue.add(_rows().iloc[3:]) == 0


def test_context_queue_rejects_unjudged_actions() -> None:
    queue = ContextQueue(maximum_rows=2)
    with pytest.raises(ValueError, match="symbolic_label"):
        queue.add(_rows().drop(columns=["symbolic_label"]))


def test_beamrider_interval_judge_assigns_one_death_label_to_cache() -> None:
    rows = _rows()
    rows.loc[1, "observed_reward"] = 44.0
    rows.loc[2, "observed_reward"] = 0.0
    rows.loc[3, "lives_after"] = 2
    labeled = label_beamrider_interval(rows)
    assert labeled["symbolic_label"].tolist() == [-1, -1, -1, -1]


def test_beamrider_interval_judge_marks_successful_cache() -> None:
    rows = _rows().iloc[:3].copy()
    labeled = label_beamrider_interval(rows)
    assert labeled["symbolic_label"].tolist() == [1, 1, 1]
