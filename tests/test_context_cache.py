import pandas as pd
import pytest

from tfm4atari.config import BeamRiderCacheConfig, load_config
from tfm4atari.context_cache import ContextQueue
from tfm4atari.games.beamrider import (
    BeamRiderRelevanceFilter,
    label_beamrider_interval,
)
from tfm4atari.pipeline import _label_teacher_symbolic_batches


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


def test_teacher_symbolic_batches_keep_every_action() -> None:
    config = load_config()
    game = config.active_games[0]
    labeled = _label_teacher_symbolic_batches(config, game, _rows())
    assert labeled["step"].tolist() == [0, 1, 2, 3]
    assert len(labeled) == len(_rows())


def test_teacher_uses_smaller_fixed_credit_windows() -> None:
    config = load_config()
    config = config.model_copy(
        update={
            "context_cache": config.context_cache.model_copy(
                update={"teacher_judgment_capacity": 2, "judgment_capacity": 4}
            )
        }
    )
    rows = _rows()
    rows["observed_reward"] = [0.0, 44.0, 0.0, 0.0]
    rows["lives_after"] = rows["lives_before"]
    labeled = _label_teacher_symbolic_batches(config, config.active_games[0], rows)
    assert labeled["symbolic_label"].tolist() == [1, 1, 0, 0]


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
