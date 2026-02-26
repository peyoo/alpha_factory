from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from alpha_factory.cli.utils import get_tushare_token, validate_date_str
from alpha_factory.data_provider.tushare_service import (
    TushareDataService,
    DataSyncError,
)


console = Console()


def sync(
    start_date: Optional[str] = typer.Option(
        None,
        "-s",
        "--start-date",
        callback=validate_date_str,
        help="开始日期，格式: YYYYMMDD；省略则执行增量更新",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "-e",
        "--end-date",
        callback=validate_date_str,
        help="结束日期，格式: YYYYMMDD，可选",
    ),
):
    """同步 Tushare 数据。

    - 不传 `start_date` 时执行日常增量更新；
    - 传入 `start_date` 时按日期区间执行全量/区间同步。
    """

    if start_date is None:
        # 增量更新路径
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
        return

    # 全量/区间同步路径（保留原有行为）
    typer.echo(f"🚀 准备同步数据: start_date={start_date}, end_date={end_date}")
    try:
        service = TushareDataService()
        service.sync_data(start_date, end_date)
        typer.echo("✅ 数据同步完成")
    except DataSyncError as e:
        console.print(f"[red]❌ 同步过程发生致命错误: {e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]❌ 初始化或同步失败: {e}[/red]")
        raise typer.Exit(code=1)


__all__ = ["sync"]
