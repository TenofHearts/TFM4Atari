import pandas as pd
import pytest

from tfm4atari.config import BeamRiderCacheConfig
from tfm4atari.context_cache import ContextQueue
from tfm4atari.games.beamrider import BeamRiderRelevanceFilter


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "online_trajectory_id": ["run", "run", "run", "run"],
            "step": [0, 1, 2, 3],
            "candidate_action": [0, 1, 2, 3],
            "observed_reward": [0.0, 0.0, 44.0, 0.0],
            "lives_before": [3, 3, 3, 3],
            "lives_after": [3, 3, 3, 2],
            "action_success": [False, True, True, False],
            "label_source": ["judge"] * 4,
            "judged_window_id": ["window"] * 4,
        }
    )


def test_beamrider_filter_runs_before_queueing() -> None:
    rule = BeamRiderRelevanceFilter(BeamRiderCacheConfig(combat_actions=(1,)))
    relevant = rule.filter(_rows())
    assert relevant["step"].tolist() == [1, 2, 3]

    queue = ContextQueue(maximum_rows=10)
    queue.add(relevant)
    assert queue.rows["step"].tolist() == [1, 2, 3]
    assert 0 not in queue.rows["step"].tolist()


def test_context_queue_is_fifo_and_deduplicates_executed_actions() -> None:
    queue = ContextQueue(maximum_rows=2)
    assert queue.add(_rows().iloc[:2]) == 2
    assert queue.add(_rows().iloc[1:]) == 2
    assert queue.rows["step"].tolist() == [2, 3]
    assert queue.add(_rows().iloc[3:]) == 0


def test_context_queue_rejects_unjudged_actions() -> None:
    queue = ContextQueue(maximum_rows=2)
    with pytest.raises(ValueError, match="action_success"):
        queue.add(_rows().drop(columns=["action_success"]))
