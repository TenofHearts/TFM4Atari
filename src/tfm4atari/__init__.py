"""TabPFN trajectory-context learning for Atari."""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Load the CLI lazily so library imports stay lightweight."""
    from tfm4atari.cli import main as cli_main

    cli_main()
