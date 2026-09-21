from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

import tfm4atari.pipeline as pipeline
from tfm4atari.config import load_config
from tfm4atari.judges import Judgment
from tfm4atari.pipeline import (
    EpisodeResult,
    _concatenate_episode_videos,
    record_learning_video,
)


def test_record_learning_video_requires_adaptive_persistent_cache() -> None:
    config = load_config(Path(__file__).parents[1] / "config.toml")
    disabled_cache = config.model_copy(
        update={
            "context_cache": config.context_cache.model_copy(
                update={"enabled": False}
            )
        }
    )
    with pytest.raises(RuntimeError, match="context_cache.enabled"):
        record_learning_video(disabled_cache)

    nonpersistent = config.model_copy(
        update={
            "video": config.video.model_copy(
                update={"persist_context_additions": False}
            )
        }
    )
    with pytest.raises(RuntimeError, match="persist_context_additions"):
        record_learning_video(nonpersistent)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_episode_videos_are_concatenated_without_reencoding(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    clips: list[Path] = []
    for index, color in enumerate(("red", "blue")):
        clip = tmp_path / f"episode-{index}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=32x32:r=10:d=0.2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(clip),
            ],
            check=True,
        )
        clips.append(clip)

    output = tmp_path / "joined.mp4"
    _concatenate_episode_videos(clips, output)

    assert output.is_file()
    assert output.stat().st_size > 0
    assert all(clip.is_file() for clip in clips)
    assert not list(tmp_path.glob("*.concat.txt"))
    assert not list(tmp_path.glob("*.partial.mp4"))


def test_learning_video_carries_context_and_checkpoints_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path(__file__).parents[1] / "config.toml")
    config = config.model_copy(
        update={
            "root": tmp_path,
            "video": config.video.model_copy(
                update={"learning_episodes": 2, "max_decisions": 4}
            ),
        }
    )
    fit_context_sizes: list[int] = []
    saved_trials: list[dict[str, object]] = []

    class FakeStore:
        policy_namespace = "outcome_conditioned_probability_sample"

        def __init__(self, _config: object) -> None:
            pass

        def read_context_cache(self, _game: str, _backend: str) -> pd.DataFrame:
            return pd.DataFrame()

        def write_context_cache(
            self, _game: str, _backend: str, _rows: pd.DataFrame
        ) -> None:
            pass

        def write_trial(
            self,
            _game: str,
            _backend: str,
            _trial_id: str,
            row: dict[str, object],
            *,
            phase: str,
        ) -> None:
            assert phase == "learning_video"
            saved_trials.append(row)

    class FakeJudge:
        def judge(self, outcome: object, _state: object) -> Judgment:
            return Judgment("fake", 1, 1.0, True, ("pass",), {})

    class FakeRecording:
        def __init__(self, *, video_folder: str, name_prefix: str, **_: object) -> None:
            self.video_folder = Path(video_folder)
            self.name_prefix = name_prefix

    def fake_fit(
        _factory: object,
        _rows: pd.DataFrame,
        *,
        additions: pd.DataFrame,
        **_: object,
    ) -> object:
        fit_context_sizes.append(len(additions))
        return object()

    def fake_run_actor(
        _config: object,
        _game: object,
        _actor: object,
        *,
        queue: object,
        environment: FakeRecording,
        seed: int,
        persist_queue: object,
        **_: object,
    ) -> EpisodeResult:
        episode_index = len(fit_context_sizes) - 1
        queue.add(
            pd.DataFrame(
                [
                    {
                        "online_trajectory_id": f"trajectory-{episode_index}",
                        "step": 0,
                        "executed_action": 1,
                        "symbolic_label": 1,
                        "desired_symbolic_label": 1,
                        "label_source": "test",
                        "judged_window_id": f"window-{episode_index}",
                    }
                ]
            )
        )
        persist_queue(queue.rows)
        environment.video_folder.mkdir(parents=True, exist_ok=True)
        (environment.video_folder / f"{environment.name_prefix}-episode-0.mp4").write_bytes(
            b"episode"
        )
        return EpisodeResult(
            initial_ram=pd.Series([0]).to_numpy(),
            episode_return=float(seed - config.video.seed + 1),
            episode_length=4,
            decision_limit=4,
            initial_lives=3,
            final_lives=3,
            positive_reward_events=1,
            terminated=False,
            truncated=False,
        )

    def fake_concatenate(_clips: list[Path], output: Path) -> None:
        output.write_bytes(b"joined")

    monkeypatch.setattr(pipeline, "DataStore", FakeStore)
    monkeypatch.setattr(pipeline, "TabPFNFactory", lambda _config: object())
    monkeypatch.setattr(
        pipeline,
        "_cold_start_context",
        lambda *_args: (pd.DataFrame(), ("teacher-a", "teacher-b")),
    )
    monkeypatch.setattr(
        pipeline, "_judge_for_game", lambda *_args: (FakeJudge(), {})
    )
    monkeypatch.setattr(
        pipeline,
        "game_module",
        lambda _config, _game: SimpleNamespace(relevance_filter=object()),
    )
    monkeypatch.setattr(pipeline.Actor, "fit", staticmethod(fake_fit))
    monkeypatch.setattr(pipeline, "build_env", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "RecordVideo", lambda _env, **kwargs: FakeRecording(**kwargs))
    monkeypatch.setattr(pipeline, "_run_actor", fake_run_actor)
    monkeypatch.setattr(pipeline, "_concatenate_episode_videos", fake_concatenate)

    result = record_learning_video(config)["BeamRider"]

    assert fit_context_sizes == [0, 1]
    assert result["episode_scores"] == [1.0, 2.0]
    assert len(saved_trials) == 2
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["scores"] == [1.0, 2.0]
    assert report["first_to_last_score_change"] == 1.0
    assert report["final_context_rows"] == 2
    assert Path(result["video"]).read_bytes() == b"joined"
