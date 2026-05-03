from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpha_factory.cli.main import app


def test_clean_removes_configured_cache_dirs(tmp_path: Path, monkeypatch):
    runner = CliRunner()

    cache_a = tmp_path / "cache_a"
    cache_b = tmp_path / "cache_b"
    cache_a.mkdir(parents=True)
    cache_b.mkdir(parents=True)
    (cache_a / "a.txt").write_text("a")
    (cache_b / "nested").mkdir(parents=True)
    (cache_b / "nested" / "b.txt").write_text("b")

    monkeypatch.setattr(
        "alpha_factory.cli.clean._repo_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "alpha_factory.cli.clean.CACHE_TARGETS",
        ("cache_a", "cache_b"),
    )

    result = runner.invoke(app, ["clean"])
    assert result.exit_code == 0, result.output
    assert "清理完成" in result.output
    assert not cache_a.exists()
    assert not cache_b.exists()


def test_clean_dry_run_keeps_cache_dirs(tmp_path: Path, monkeypatch):
    runner = CliRunner()

    cache_dir = tmp_path / "cache_preview"
    cache_dir.mkdir(parents=True)
    (cache_dir / "preview.txt").write_text("preview")

    monkeypatch.setattr(
        "alpha_factory.cli.clean._repo_root",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "alpha_factory.cli.clean.CACHE_TARGETS",
        ("cache_preview",),
    )

    result = runner.invoke(app, ["clean", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "预览完成" in result.output
    assert cache_dir.exists()
