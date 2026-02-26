from __future__ import annotations

import typer

from alpha_factory.cli import main


def test_app_exists_and_is_typer():
    assert hasattr(main, "app")
    assert isinstance(main.app, typer.Typer)
