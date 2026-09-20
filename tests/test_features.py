import numpy as np

from tfm4atari.features import (
    ACTION_FEATURE_COLUMNS,
    DESIRED_SYMBOLIC_LABEL,
    DefaultRamFeatureExtractor,
)


def test_state_features_include_requested_symbolic_outcome() -> None:
    extractor = DefaultRamFeatureExtractor()
    features = extractor.state_features(
        np.arange(128, dtype=np.uint8),
        None,
        previous_action=-1,
        previous_reward=0.0,
        lives=3,
        episode_progress=0.0,
        desired_symbolic_label=1,
    )
    assert features[DESIRED_SYMBOLIC_LABEL] == 1
    assert list(features) == list(ACTION_FEATURE_COLUMNS)


def test_symbolic_outcome_must_be_ternary() -> None:
    extractor = DefaultRamFeatureExtractor()
    try:
        extractor.state_features(
            np.zeros(128, dtype=np.uint8),
            None,
            previous_action=-1,
            previous_reward=0.0,
            lives=3,
            episode_progress=0.0,
            desired_symbolic_label=2,
        )
    except ValueError as error:
        assert "-1, 0, or 1" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid symbolic label to fail")
