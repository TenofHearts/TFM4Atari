import numpy as np
import pandas as pd

from tfm4atari.features import (
    ACTION_FEATURE_COLUMNS,
    ACTION_TARGET_COLUMN,
    DESIRED_SYMBOLIC_LABEL,
)
from tfm4atari.models import Actor, stratified_context


class RecordingClassifier:
    classes_ = np.array([False, True])

    def fit(self, x, y):
        self.x = x.copy()
        self.y = y.copy()

    def predict_proba(self, x):
        self.query = x.copy()
        return np.tile([0.5, 0.5], (len(x), 1))


class RecordingFactory:
    def classifier(self, categorical_columns=()):
        self.model = RecordingClassifier()
        return self.model


def _action_rows(count: int, marker: int) -> pd.DataFrame:
    rows = pd.DataFrame(0, index=range(count), columns=ACTION_FEATURE_COLUMNS)
    rows["step"] = range(count)
    rows[ACTION_TARGET_COLUMN] = [index % 2 for index in range(count)]
    rows[DESIRED_SYMBOLIC_LABEL] = [index % 3 - 1 for index in range(count)]
    rows["ram_000"] = marker
    return rows


def test_actor_context_reserves_rl_additions_and_cold_start() -> None:
    factory = RecordingFactory()
    Actor.fit(
        factory,
        _action_rows(10, marker=1),
        budget=6,
        seed=1,
        additions=_action_rows(3, marker=2),
        minimum_base_rows=4,
    )
    assert len(factory.model.x) == 6
    assert factory.model.x["ram_000"].value_counts().to_dict() == {1: 4, 2: 2}


def test_cold_start_sampling_preserves_both_teacher_trajectories() -> None:
    first = _action_rows(10, marker=1).assign(trajectory_id="first")
    second = _action_rows(10, marker=2).assign(trajectory_id="second")
    context = stratified_context(
        pd.concat([first, second], ignore_index=True), budget=8, seed=1
    )
    assert set(context["trajectory_id"]) == {"first", "second"}


def test_context_sampling_preserves_action_frequency() -> None:
    common = _action_rows(900, marker=1)
    common[ACTION_TARGET_COLUMN] = 0
    rare = _action_rows(100, marker=1)
    rare[ACTION_TARGET_COLUMN] = 1
    rows = pd.concat([common, rare], ignore_index=True).assign(
        trajectory_id="teacher", desired_symbolic_label=1
    )
    context = stratified_context(rows, budget=200, seed=1)
    assert len(context) == 200
    assert context[ACTION_TARGET_COLUMN].value_counts().to_dict() == {0: 180, 1: 20}


def test_actor_queries_for_success_label_one() -> None:
    factory = RecordingFactory()
    actor = Actor.fit(factory, _action_rows(10, marker=1), budget=10, seed=1)
    actor.choose_action(
        np.zeros(128, dtype=np.uint8),
        None,
        action_count=2,
        previous_action=-1,
        previous_reward=0.0,
        lives=3,
        episode_progress=0.0,
    )
    assert factory.model.query[DESIRED_SYMBOLIC_LABEL].tolist() == [1]
