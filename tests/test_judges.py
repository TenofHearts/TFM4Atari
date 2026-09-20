from tfm4atari.config import (
    BeamRiderSymbolicJudgeConfig,
    ReturnQuantileJudgeConfig,
)
from tfm4atari.games.beamrider import BeamRiderSymbolicJudge
from tfm4atari.judges import ReturnQuantileJudge, TrajectoryOutcome


def _outcome(score: float, *, decisions: int = 64, final_lives: int = 3):
    return TrajectoryOutcome(
        game="BeamRider",
        episode_return=score,
        decisions=decisions,
        decision_limit=64,
        initial_lives=3,
        final_lives=final_lives,
        positive_reward_events=int(score > 0),
        terminated=False,
        truncated=False,
    )


def test_beamrider_symbolic_judge_requires_progress_and_survival() -> None:
    judge = BeamRiderSymbolicJudge(BeamRiderSymbolicJudgeConfig())
    assert judge.judge(_outcome(44), {}).success
    failed = judge.judge(_outcome(0, decisions=32, final_lives=1), {})
    assert not failed.success
    assert "minimum_score=fail" in failed.reasons
    assert "survival=fail" in failed.reasons


def test_return_quantile_judge_is_pluggable() -> None:
    judge = ReturnQuantileJudge(ReturnQuantileJudgeConfig(success_quantile=0.5))
    state = judge.calibrate([_outcome(0), _outcome(100)])
    assert state["threshold"] == 50
    assert judge.judge(_outcome(100), state).success
    assert not judge.judge(_outcome(0), state).success
