from __future__ import annotations

import shutil
from pathlib import Path
from typing import Final

import typer

CACHE_TARGETS: Final[tuple[str, ...]] = (
    "data/tmp_cache",
    "output/tmp_data",
    "output/pip-logs",
    "output/conda-logs",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _count_entries(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for _ in path.rglob("*"))


def clean(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="仅预览将清理的缓存目录，不实际删除",
    ),
) -> None:
    """清理系统中的临时缓存目录。"""

    root = _repo_root()
    targets = [root / rel for rel in CACHE_TARGETS]

    removed_dirs = 0
    removed_entries = 0

    for target in targets:
        entries = _count_entries(target)
        if entries == 0:
            typer.echo(f"⏭ 跳过: {target.relative_to(root)} (空或不存在)")
            continue

        removed_dirs += 1
        removed_entries += entries
        typer.echo(f"🧹 清理: {target.relative_to(root)} (条目数: {entries})")

        if dry_run:
            continue

        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)

    mode_text = "预览完成" if dry_run else "清理完成"
    typer.echo(
        f"✅ {mode_text}: 命中目录 {removed_dirs} 个，涉及条目 {removed_entries} 个。"
    )


__all__ = ["clean"]
