from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from alpha_factory.cli.utils import get_tushare_token, validate_date_str
from alpha_factory.data_provider.tushare_service import TushareDataService, DataSyncError


console = Console()

data_app = typer.Typer(help="数据子命令：包含数据同步/更新相关的命令。")


@data_app.command()
def sync(
    start_date: str = typer.Argument(
        ..., callback=validate_date_str, help="开始日期，格式: YYYYMMDD"
    ),
    end_date: Optional[str] = typer.Option(
        None, callback=validate_date_str, help="结束日期，格式: YYYYMMDD，可选"
    ),
):
    """按日期区间全量同步 Tushare 数据（将异常直接抛给用户处理）。"""
    token = get_tushare_token()
    if not token:
        console.print(
            "[red]❌ 未检测到 TUSHARE_TOKEN（settings.TUSHARE_TOKEN 为空）。请在环境变量或配置中设置后重试。[/red]"
        )
        raise typer.Exit(code=1)

    typer.echo(f"🚀 准备同步数据: start_date={start_date}, end_date={end_date}")

    try:
        service = TushareDataService()
        service.sync_data(start_date, end_date)
        typer.echo("✅ 数据同步完成")
    except DataSyncError as e:
        # 将致命的同步错误直接交给用户处理（打印并退出）
        console.print(f"[red]❌ 同步过程发生致命错误: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        # 其他异常也直接反馈给用户，不做自动修复
        console.print(f"[red]❌ 初始化或同步失败: {e}[/red]")
        raise typer.Exit(code=1)


@data_app.command()
def update():
    """执行日常增量更新（将异常直接抛给用户处理）。"""
    token = get_tushare_token()
    if not token:
        console.print(
            "[red]❌ 未检测到 TUSHARE_TOKEN（settings.TUSHARE_TOKEN 为空）。请在环境变量或配置中设置后重试。[/red]"
        )
        raise typer.Exit(code=1)

    typer.echo("🔄 准备执行日常增量更新")

    try:
        service = TushareDataService()
        service.daily_update()
        typer.echo("✅ 增量更新完成")
    except DataSyncError as e:
        console.print(f"[red]❌ 增量更新过程中发生致命错误: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]❌ 初始化或更新失败: {e}[/red]")
        raise typer.Exit(code=1)


__all__ = ["data_app"]
