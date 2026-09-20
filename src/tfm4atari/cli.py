"""Small config-only command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from tfm4atari.config import ProjectConfig, load_config
from tfm4atari.pipeline import (
    bootstrap_judges,
    collect,
    evaluate,
    fetch_teachers,
    format_result,
    play,
    preflight,
    prepare_tabpfn,
    record_teacher_video,
    record_video,
    run_pipeline,
)

COMMANDS: dict[str, Callable[[ProjectConfig], Any]] = {
    "preflight": preflight,
    "fetch-teachers": fetch_teachers,
    "prepare-tabpfn": prepare_tabpfn,
    "collect": collect,
    "bootstrap-judges": bootstrap_judges,
    "play": play,
    "evaluate": evaluate,
    "record-video": record_video,
    "record-teacher-video": record_teacher_video,
    "pipeline": run_pipeline,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tfm4atari",
        description="TabPFN Atari pipeline; all settings come from ./config.toml",
    )
    parser.add_argument("command", choices=tuple(COMMANDS))
    args = parser.parse_args()
    config = load_config()
    print(format_result(COMMANDS[args.command](config)))
