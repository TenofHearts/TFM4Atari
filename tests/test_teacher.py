from types import SimpleNamespace

import numpy as np
import torch

from tfm4atari.teacher import TEACHER_BACKENDS, RLZooPPOTeacher, RLZooQRDQNTeacher


class FakePolicy:
    def obs_to_tensor(self, observation):
        return torch.as_tensor(observation), None

    def get_distribution(self, tensor):
        del tensor
        categorical = SimpleNamespace(
            probs=torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float32)
        )
        return SimpleNamespace(distribution=categorical)


class FakeQRDQN:
    policy = FakePolicy()

    def quantile_net(self, tensor):
        del tensor
        return torch.tensor([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]], dtype=torch.float32)


def test_teacher_backend_registry_is_complete() -> None:
    assert set(TEACHER_BACKENDS) == {"dqn", "qrdqn", "ppo"}


def test_qrdqn_uses_mean_quantile_action_values(tmp_path) -> None:
    teacher = RLZooQRDQNTeacher(FakeQRDQN(), tmp_path / "teacher.zip", "qrdqn")
    scores = teacher.action_scores(np.zeros((1, 1)))
    assert scores.tolist() == [2.0, 3.0, 4.0]


def test_ppo_uses_policy_probabilities_as_preferences(tmp_path) -> None:
    model = SimpleNamespace(policy=FakePolicy())
    teacher = RLZooPPOTeacher(model, tmp_path / "teacher.zip", "ppo")
    scores = teacher.action_scores(np.zeros((1, 1)))
    assert np.allclose(scores, [0.1, 0.7, 0.2])
