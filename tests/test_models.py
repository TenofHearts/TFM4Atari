import numpy as np
import pandas as pd

from tfm4atari.features import ACTION_FEATURE_COLUMNS
from tfm4atari.models import Actor, stratified_context


class RecordingClassifier:
    classes_ = np.array([False, True])

    def fit(self, x, y):
        self.x = x.copy()
        self.y = y.copy()

    def predict_proba(self, x):
        return np.tile([0.5, 0.5], (len(x), 1))


class RecordingFactory:
    def classifier(self, categorical_columns=()):
        self.model = RecordingClassifier()
        return self.model


def _action_rows(count: int, marker: int) -> pd.DataFrame:
    rows = pd.DataFrame(0, index=range(count), columns=ACTION_FEATURE_COLUMNS)
    rows["step"] = range(count)
    rows["candidate_action"] = [index % 2 for index in range(count)]
    rows["action_success"] = [bool(index % 2) for index in range(count)]
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
