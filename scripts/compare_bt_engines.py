"""
比较 backtest_daily_evolving 与 backtest_quick_daily 结果一致性。
用法: uv run python scripts/compare_bt_engines.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from rich.console import Console
from rich.table import Table

from alpha_factory.config.strategy import StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.evaluation.backtest.daily_evolving import backtest_daily_evolving
from alpha_factory.evaluation.backtest.quick_daily import backtest_quick_daily
from alpha_factory.cli.opt import _resolve_pool
from alpha_factory.utils.schema import F

console = Console()

YAML_PATH = Path("output/strategies/s2.yaml")
START_DATE = "20190101"
END_DATE = None

_EXEC_PRICE_MAP = {"open": F.OPEN, "close": F.CLOSE, "vwap": F.VWAP}


def main() -> None:
    cfg = StrategyConfig.from_yaml(YAML_PATH)
    pool_instance = _resolve_pool(cfg.pool)
    exec_price_col = _EXEC_PRICE_MAP.get(cfg.exe_price.strip().lower(), F.VWAP)
    factor_col = cfg.ranks[0].name  # s2 是单因子

    console.print(
        f"[bold cyan]📦 加载数据[/bold cyan] factor={factor_col} | "
        f"pool={cfg.pool} | hold={cfg.hold_num} | sell_rank={cfg.sell_rank} | "
        f"exec_price={exec_price_col}"
    )

    dp = DataProvider()
    all_exprs = cfg.ranked_factor_exprs + cfg.get_condition_exprs()
    lf = dp.load_pool_data(
        pool_instance,
        START_DATE,
        END_DATE,
        exprs=all_exprs,
        actions=cfg.build_actions(),
    )
    # 两个引擎共享同一份原始 df（均在内部各自排名）
    df = lf.collect()
    console.print(f"数据行数: {len(df):,}  列: {df.columns}")

    # ---------- 引擎 A：daily_evolving ----------
    console.print("\n[bold green]▶ 运行 backtest_daily_evolving ...[/bold green]")
    result_a = backtest_daily_evolving(
        df_input=df,
        factor_col=factor_col,
        n_buy=cfg.hold_num,
        sell_rank=cfg.sell_rank,
        cost_rate=cfg.cost,
        exec_price=exec_price_col,
        ascending=False,
    )
    daily_a = result_a["daily_results"]
    trades_a = result_a["trade_details"]

    # ---------- 引擎 B：quick_daily ----------
    console.print("\n[bold yellow]▶ 运行 backtest_quick_daily ...[/bold yellow]")
    result_b = backtest_quick_daily(
        df_input=df,
        factor_col=factor_col,
        n_buy=cfg.hold_num,
        sell_rank=cfg.sell_rank,
        cost_rate=cfg.cost,
        exec_price=exec_price_col,
        ascending=False,
    )
    daily_b = result_b["daily_results"]
    trades_b = result_b["trade_details"]

    # ---------- 对齐日期后比较 NAV ----------
    nav_a = daily_a["NAV"].to_numpy()
    nav_b = daily_b["NAV"].to_numpy()

    # 对齐行数（两引擎日期应完全相同）
    n = min(len(nav_a), len(nav_b))
    nav_a, nav_b = nav_a[:n], nav_b[:n]

    diff = nav_a - nav_b
    abs_diff = np.abs(diff)
    rel_diff = abs_diff / np.maximum(np.abs(nav_a), 1e-10)

    final_nav_a = float(nav_a[-1])
    final_nav_b = float(nav_b[-1])

    # ---------- 汇总输出 ----------
    table = Table(title="引擎对比摘要", header_style="bold magenta")
    table.add_column("指标", style="cyan", min_width=22)
    table.add_column("daily_evolving", justify="right", min_width=14)
    table.add_column("quick_daily (JIT)", justify="right", min_width=14)
    table.add_column("差值", justify="right", min_width=14)

    def fmt(v: float, pct: bool = False) -> str:
        return f"{v:.4%}" if pct else f"{v:.6f}"

    table.add_row(
        "最终 NAV", fmt(final_nav_a), fmt(final_nav_b), fmt(final_nav_a - final_nav_b)
    )
    table.add_row("NAV 最大绝对差", "", "", fmt(float(abs_diff.max())))
    table.add_row("NAV 最大相对差", "", "", fmt(float(rel_diff.max()), pct=True))
    table.add_row(
        "NAV 均方根误差 (RMSE)", "", "", fmt(float(np.sqrt(np.mean(diff**2))))
    )
    table.add_row(
        "交易天数",
        str(len(daily_a)),
        str(len(daily_b)),
        str(len(daily_a) - len(daily_b)),
    )
    table.add_row(
        "交易笔数",
        str(len(trades_a)),
        str(len(trades_b)),
        str(len(trades_a) - len(trades_b)),
    )

    console.print(table)

    # ---------- 诊断：差异最大的 10 天 ----------
    if float(abs_diff.max()) > 1e-6:
        console.print("\n[bold red]⚠ 发现差异，列出差值最大的 10 天：[/bold red]")
        dates_a = daily_a[F.DATE].to_list()[:n]
        top_idx = np.argsort(abs_diff)[-10:][::-1]
        diag = pl.DataFrame(
            {
                "date": [dates_a[i] for i in top_idx],
                "NAV_evolving": nav_a[top_idx],
                "NAV_quick": nav_b[top_idx],
                "abs_diff": abs_diff[top_idx],
                "rel_diff_%": rel_diff[top_idx] * 100,
            }
        )
        console.print(diag)

        # ---------- 细查：数据准备差异 ----------
        console.print("\n[bold]📋 检查数据准备层差异[/bold]")

        # quick_daily 的 RANK 是在 backtest 内部算的，这里单独重算看列名是否能正常访问
        try:
            test_lf = (
                df.lazy()
                .with_columns(
                    pl.when(pl.col(F.POOL_MASK))
                    .then(pl.col(factor_col))
                    .otherwise(None)
                    .rank(descending=True, method="ordinal")
                    .over(F.DATE)
                    .fill_null(999999)
                    .alias("_TEST_RANK")
                )
                .select(
                    [
                        F.DATE,
                        F.ASSET,
                        "_TEST_RANK",
                        F.CLOSE,
                        exec_price_col,
                        F.IS_SUSPENDED,
                        F.IS_UP_LIMIT,
                        F.IS_DOWN_LIMIT,
                    ]
                )
            )
            test_df = test_lf.collect()
            console.print("[green]✅ 列名访问正常，样本行:[/green]")
            console.print(test_df.head(3))
        except Exception as e:
            console.print(f"[red]❌ 列名访问报错: {e}[/red]")

    else:
        console.print(
            "\n[bold green]✅ 两引擎 NAV 完全一致（差值 < 1e-6）[/bold green]"
        )


if __name__ == "__main__":
    main()
