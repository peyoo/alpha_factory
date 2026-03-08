from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
import sys
from typing import Any, Optional

import typer
from rich.console import Console

from alpha_factory.config.base import settings
from alpha_factory.data_provider.pool import MainSmallPool

console = Console()


class PoolUniverseEnum(Enum):
    main_small = MainSmallPool


def get_tushare_token() -> str:
    """从全局配置返回 TUSHARE token。

    返回:
        str: settings 中的 TUSHARE_TOKEN 字符串（可能为占位符或空字符串）。

    说明:
        不在此处强制退出或抛出异常，调用方可根据返回值决定是否中断。
    """
    # 自检：确保 settings 包含该属性
    assert hasattr(settings, "TUSHARE_TOKEN"), "配置缺失: settings.TUSHARE_TOKEN"

    token = settings.TUSHARE_TOKEN
    # 将 None 统一为空字符串，便于上层判断
    return token or ""


def validate_date_str(
    ctx: typer.Context, param: Any, value: Optional[str]
) -> Optional[str]:
    """Typer 回调函数，用于验证日期字符串格式为 YYYYMMDD。

    如果格式不正确，使用 rich 打印错误并退出进程（sys.exit(1)），以配合 CLI 的交互习惯。

    该函数签名兼容 Typer 的 callback 要求： (ctx, param, value) -> value
    """
    if value is None:
        return None

    if not isinstance(value, str):
        console.print(
            f"[red]❌ 参数 `{getattr(param, 'name', str(param))}` 必须为字符串 (YYYYMMDD)。[/red]"
        )
        sys.exit(1)

    try:
        datetime.strptime(value, "%Y%m%d")
        return value
    except Exception:
        console.print(
            f"[red]❌ 参数 `{getattr(param, 'name', str(param))}` 的日期格式不正确，期望: YYYYMMDD，收到: {value}[/red]"
        )
        sys.exit(1)


__all__ = [
    "get_tushare_token",
    "validate_date_str",
    "PoolUniverseEnum",
    "resolve_yaml_path",
]


def resolve_yaml_path(yaml_file: Path) -> Path:
    """将 YAML 文件路径解析为绝对路径。

    解析优先级（优先级升序）：

    1. 已是绝对路径 → 原样返回。
    2. cwd / yaml_file 存在 → 返回（向前兼容）。
    3. settings.STRATEGY_DIR / yaml_file 存在 → 返回。
    4. 备选 → 返回 cwd / yaml_file（后续会报文件不存在）。
    """
    p = Path(yaml_file)
    if p.is_absolute():
        return p
    as_cwd = Path.cwd() / p
    if as_cwd.exists():
        return as_cwd
    as_strategy = settings.STRATEGY_DIR / p
    if as_strategy.exists():
        return as_strategy
    return as_cwd
