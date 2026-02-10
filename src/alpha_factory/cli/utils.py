from __future__ import annotations

from datetime import datetime
import sys
from typing import Any, Optional

import typer
from rich.console import Console

from alpha_factory.config.base import settings

console = Console()


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


def validate_date_str(ctx: typer.Context, param: Any, value: Optional[str]) -> Optional[str]:
    """Typer 回调函数，用于验证日期字符串格式为 YYYYMMDD。

    如果格式不正确，使用 rich 打印错误并退出进程（sys.exit(1)），以配合 CLI 的交互习惯。

    该函数签名兼容 Typer 的 callback 要求： (ctx, param, value) -> value
    """
    if value is None:
        return None

    if not isinstance(value, str):
        console.print(f"[red]❌ 参数 `{getattr(param, 'name', str(param))}` 必须为字符串 (YYYYMMDD)。[/red]")
        sys.exit(1)

    try:
        datetime.strptime(value, "%Y%m%d")
        return value
    except Exception:
        console.print(
            f"[red]❌ 参数 `{getattr(param, 'name', str(param))}` 的日期格式不正确，期望: YYYYMMDD，收到: {value}[/red]"
        )
        sys.exit(1)


__all__ = ["get_tushare_token", "validate_date_str"]

