import numpy as np

from tfm4atari.features import (
    ACTION_FEATURE_COLUMNS,
    DefaultRamFeatureExtractor,
    make_action_rows,
    q_success_labels,
)


def test_q_labels_preserve_cutoff_ties() -> None:
    result = q_success_labels(
        np.array([0.0, 1.0, 2.0, 2.0]), quantile=0.75, epsilon=1e-6
    )
    assert result is not None
    advantages, labels = result
    assert np.isfinite(advantages).all()
    assert labels.tolist() == [0, 0, 1, 1]


def test_degenerate_q_vector_is_excluded() -> None:
    assert q_success_labels(np.ones(4), quantile=0.75, epsilon=1e-6) is None


def test_action_rows_are_bounded_and_label_every_action() -> None:
    rows = make_action_rows(
        DefaultRamFeatureExtractor(),
        np.arange(128, dtype=np.uint8),
        None,
        np.array([-1.0, 0.0, 1.0, 2.0]),
        previous_action=-1,
        previous_reward=0.0,
        lives=3,
        episode_progress=0.0,
        success_quantile=0.75,
        q_epsilon=1e-6,
    )
    assert len(ACTION_FEATURE_COLUMNS) == 262
    assert len(rows) == 4
    assert rows["first_step"].eq(1).all()
    assert rows["delta_000"].eq(0).all()
    assert rows["action_success"].sum() == 1
