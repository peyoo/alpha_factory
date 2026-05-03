"""quant trades —— 策略潜在交易分析命令

基于策略 YAML 生成所有潜在交易，并运行两种 LightGBM 分析：

* **filter 分析**（排雷）：坏交易打标（1），找出买入时刻导致亏损的特征 →
  建议加入 ``pool_mask`` / ``not_buy_able``。
* **rank 分析**（选优）：以 ``pnl_ret`` 为目标做回归，找出预测高收益的特征 →
  建议强化 ``ranks`` 权重。

两种分析均只使用买入时刻快照特征（无前瞻），特征由模块级 ``FEATURE_EXPRS``
定义或通过 ``--feature`` 参数传入，支持 ``name=expr`` 命名或按顺序自动命名。

示例::

    quant trades -y output/main_small_pool/s1.yaml
    quant trades -y s1 -s 20210101 --end 20241231 --no-report
    quant trades -y s1 --bad-threshold -0.03 --save-trades trades.csv
    quant trades -y s1 --feature "mom5=CLOSE/CLOSE.shift(5)-1" --feature TOTAL_MV
"""

from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.cli._loader import resolve_pool
from alpha_factory.cli._trades_html import _generate_html_report
from alpha_factory.cli.opt import _COMPOSITE_COL
from alpha_factory.cli.utils import resolve_yaml_path
from alpha_factory.config.strategy import StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.utils.schema import F, RANK_SENTINEL

console = Console()

# ---------------------------------------------------------------------------
# 特征表达式（买入时刻快照）
# 格式："name = expr" 明确命名；无等号则按顺序自动命名 feature_0, feature_1, ...
# ---------------------------------------------------------------------------
FEATURE_EXPRS: list[str] = [
    # "total_mv = TOTAL_MV",
    "bias20 = CLOSE/ts_mean(CLOSE, 20)-1",
    # "bias120 = CLOSE/ts_mean(CLOSE, 120)-1",
    # "bias200 = CLOSE/ts_mean(CLOSE, 200)-1",
    # "pe = PE",
    # "pb = PB",
    # "turnover_rate = TURNOVER_RATE",
    # "open_ = OPEN",
    # "close = CLOSE",
    # "amount = AMOUNT",
    # "volume = VOLUME",
    # "circ_mv = CIRC_MV",
    # "vwap = VWAP",
    # "ret = RET",
]


# ---------------------------------------------------------------------------
# 特征表达式解析
# ---------------------------------------------------------------------------


def _parse_feature_exprs(raw: list[str]) -> tuple[list[str], list[str]]:
    """解析特征表达式列表，返回 (col_names, codegen_exprs)。

    规则：
    - ``"name = expr"`` 格式 → col_name=name，codegen_expr="name = expr"（规范化空格）
    - 无 ``=`` 的裸表达式 → 按顺序自动命名 ``feature_0, feature_1, ...``
    """
    col_names: list[str] = []
    codegen_exprs: list[str] = []
    auto_idx = 0
    for s in raw:
        m = re.match(r"^(\w+)\s*=\s*(.+)$", s.strip(), re.DOTALL)
        if m:
            name = m.group(1).strip()
            body = m.group(2).strip()
            col_names.append(name)
            codegen_exprs.append(f"{name} = {body}")
        else:
            name = f"feature_{auto_idx}"
            auto_idx += 1
            col_names.append(name)
            codegen_exprs.append(f"{name} = {s.strip()}")
    return col_names, codegen_exprs


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def quant_trades(
    yaml_file: Path = typer.Option(
        ...,
        "-y",
        "--yaml",
        help="策略 YAML 文件路径（StrategyConfig 格式）",
    ),
    start_date: str = typer.Option(
        "20190101", "-s", "--start-date", help="数据起始日期 YYYYMMDD"
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end", "--end-date", help="数据结束日期 YYYYMMDD（默认至最新）"
    ),
    bad_threshold: float = typer.Option(
        -0.05,
        "--bad-threshold",
        help="filter 分析坏交易阈值：pnl_ret <= 该值为坏交易（label=1）。默认 0.0。",
    ),
    test_ratio: float = typer.Option(
        0.2,
        "--test-ratio",
        min=0.05,
        max=0.5,
        help="按时间顺序划分的测试集比例（默认 0.2）",
    ),
    save_trades: Optional[Path] = typer.Option(
        None,
        "--save-trades",
        help="保存带标注的交易明细（.csv 或 .parquet）",
        show_default=False,
    ),
    report: bool = typer.Option(
        True, "--report/--no-report", help="是否生成 HTML 报告并打开"
    ),
    full_diag: bool = typer.Option(
        False,
        "--full-diag/--no-full-diag",
        help="开启全量诊断模式：额外计算 SHAP Interaction 图（计算量较大）",
    ),
    run_rank: bool = typer.Option(
        False,
        "--rank/--no-rank",
        help="是否执行 rank（选优）分析（默认关闭）",
    ),
    feature_exprs: Optional[list[str]] = typer.Option(
        None,
        "--feature",
        help=(
            "特征表达式，可多次传入。格式：name=expr 或裸表达式（按顺序自动命名 feature_N）。"
            "不传则使用模块默认 FEATURE_EXPRS。"
        ),
        show_default=False,
    ),
):
    """
    策略潜在交易分析：生成所有交易 → 建立买入特征 → filter（排雷）+ rank（选优）双模型分析。

    \\b
    示例:
      quant trades -y output/main_small_pool/s1.yaml
      quant trades -y s1 -s 20210101 --bad-threshold -0.02
    """
    # --- 加载策略配置 ---
    yaml_file = resolve_yaml_path(yaml_file)
    try:
        cfg = StrategyConfig.from_yaml(yaml_file)
    except Exception as exc:
        typer.echo(f"❌ 加载策略配置失败: {exc}", err=True)
        raise typer.Exit(code=1)

    if not cfg.ranks:
        typer.echo("❌ YAML ranks 列表为空", err=True)
        raise typer.Exit(code=1)

    try:
        pool_instance = resolve_pool(cfg.pool)
    except ValueError as exc:
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(code=1)

    console.print(
        f"[bold cyan]📦 策略[/bold cyan] {cfg.name!r} | pool={cfg.pool} | "
        f"buy_rank={cfg.buy_rank} | sell_rank={cfg.sell_rank} | "
        f"{start_date} ~ {end_date or '最新'}"
    )

    # --- 加载数据（与 quant bt 完全一致）---
    dp = DataProvider()
    all_exprs = cfg.ranked_factor_exprs + cfg.get_condition_exprs()
    lf = dp.load_pool_data(
        pool_instance,
        start_date,
        end_date,
        exprs=all_exprs,
        actions=cfg.build_actions(),
    )
    df = lf.collect()

    # --- 解析并计算特征因子 ---
    effective_exprs = feature_exprs or FEATURE_EXPRS
    feat_col_names, feat_codegen_exprs = _parse_feature_exprs(effective_exprs)
    if feat_codegen_exprs:
        console.print(f"[dim]计算特征因子：{feat_col_names}...[/dim]")
        df = dp.eval_exprs_on_df(df, feat_codegen_exprs)

    factor_col = cfg.ranks[0].name if len(cfg.ranks) == 1 else _COMPOSITE_COL

    # --- 生成潜在交易（仅已平仓）---
    console.print("[dim]生成潜在交易...[/dim]")
    trades = cfg.generate_potential_trades(
        df, factor_col, ascending=False, include_open=False
    )

    if len(trades) == 0:
        console.print("[yellow]⚠ 未找到任何已平仓交易，无法继续分析[/yellow]")
        raise typer.Exit(code=0)

    console.print(f"[green]✓[/green] 共 {len(trades)} 笔已平仓交易")

    # --- 特征拼接（买入时刻快照）---
    trades_feat = _attach_buy_features(trades, df, factor_col, feat_col_names)

    # 丢弃特征有缺失的行（只对 DataFrame 中实际存在的列做 subset，避免 ColumnNotFoundError）
    # existing_feat_cols = [c for c in FEATURE_COLS if c in trades_feat.columns]
    # trades_feat = trades_feat.drop_nulls(subset=existing_feat_cols)
    # if len(trades_feat) < 20:
    #     console.print("[yellow]⚠ 有效样本不足 20 条，无法训练模型[/yellow]")
    #     raise typer.Exit(code=0)

    if save_trades is not None:
        _save_df(trades_feat, save_trades)

    # --- 两种分析 ---
    filter_result = _run_filter_analysis(
        trades_feat, bad_threshold, test_ratio, feat_col_names
    )
    rank_result = (
        _run_rank_analysis(trades_feat, test_ratio, feat_col_names)
        if run_rank
        else None
    )

    # --- 终端打印摘要 ---
    _print_analysis_summary(trades_feat, bad_threshold, filter_result, rank_result)

    # --- HTML 报告 ---
    if report:
        if full_diag:
            console.print(
                "[dim]⚙ full-diag 模式：将额外计算 SHAP Interaction（约需 30-120 秒）...[/dim]"
            )
        try:
            filter_path, rank_path = _generate_html_report(
                cfg, trades_feat, bad_threshold, filter_result, rank_result, full_diag
            )
            console.print(
                f"[bold green]📄 Filter 报告（排雷）[/bold green] → {filter_path}"
            )
            if rank_path is not None:
                console.print(
                    f"[bold green]📄 Rank 报告（选优）[/bold green] → {rank_path}"
                )
            webbrowser.open(filter_path.as_uri())
            if rank_path is not None:
                webbrowser.open(rank_path.as_uri())
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]⚠ 报告生成失败: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# 特征拼接
# ---------------------------------------------------------------------------


def _attach_buy_features(
    trades: pl.DataFrame,
    df: pl.DataFrame,
    factor_col: str,
    feature_col_names: list[str],
) -> pl.DataFrame:
    """将买入日快照特征 join 到交易表。"""
    # 从 df 计算 RANK（与 generate_potential_trades 内部一致，直接复用已计算的列）
    # df 已经过 build_actions，POOL_MASK 列存在
    rank_df = df.with_columns(
        pl.when(pl.col(F.POOL_MASK))
        .then(pl.col(factor_col))
        .otherwise(None)
        .rank(descending=True, method="random")
        .over(F.DATE)
        .fill_null(RANK_SENTINEL)
        .alias("_RANK_SNAP")
    )

    snap_cols = [c for c in feature_col_names if c in df.columns]
    snap_df = rank_df.select(
        [F.DATE, F.ASSET, pl.col("_RANK_SNAP").alias("rank_val")]
        + [pl.col(c) for c in snap_cols]
    )

    result = trades.join(
        snap_df,
        left_on=[F.ASSET, "buy_date"],
        right_on=[F.ASSET, F.DATE],
        how="left",
    ).with_columns(
        pl.col("buy_date").dt.month().cast(pl.Int32).alias("month"),
        pl.col("buy_date").dt.weekday().cast(pl.Int32).alias("weekday"),
    )

    return result


# ---------------------------------------------------------------------------
# filter 分析（排雷：坏交易 label=1）
# ---------------------------------------------------------------------------


def _run_filter_analysis(
    trades_feat: pl.DataFrame,
    bad_threshold: float,
    test_ratio: float,
    feature_col_names: list[str],
) -> dict:
    """LGBMClassifier 分析坏交易特征。"""
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    X, y_raw, feat_names, buy_dates = _prepare_xy(trades_feat, feature_col_names)
    y = (y_raw <= bad_threshold).astype(int)  # 1 = 坏交易

    split = max(1, int(len(X) * (1 - test_ratio)))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    # numpy 版本供 SHAP / numpy 运算使用
    X_np = X.to_numpy()
    X_test_np = X_np[split:]

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    metrics: dict = {
        "model": model,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_names": feat_names,
        "X_full": X_np,
        "y_full": y_raw,
        "buy_dates_full": buy_dates,
        "pnl_test": y_raw[split:],
    }

    if len(X_test) > 0 and len(np.unique(y_test)) > 1:
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        metrics["auc"] = float(roc_auc_score(y_test, proba))
        metrics["precision"] = float(precision_score(y_test, pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, pred, zero_division=0))
    else:
        metrics["auc"] = metrics["precision"] = metrics["recall"] = float("nan")

    # SHAP（传 numpy array，explainer 已知列名无需 DataFrame）
    metrics["shap_explainer"], metrics["shap_values"], metrics["X_test_np"] = (
        _compute_shap(model, X_train.to_numpy(), X_test_np)
    )

    return metrics


# ---------------------------------------------------------------------------
# rank 分析（选优：pnl_ret 回归）
# ---------------------------------------------------------------------------


def _run_rank_analysis(
    trades_feat: pl.DataFrame,
    test_ratio: float,
    feature_col_names: list[str],
) -> dict:
    """LGBMRegressor 分析高收益特征。"""
    import numpy as np
    from lightgbm import LGBMRegressor
    from scipy.stats import spearmanr

    X, y, feat_names, buy_dates = _prepare_xy(trades_feat, feature_col_names)

    split = max(1, int(len(X) * (1 - test_ratio)))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]
    # numpy 版本供 SHAP / numpy 运算使用
    X_np = X.to_numpy()
    X_test_np = X_np[split:]

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    metrics: dict = {
        "model": model,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_names": feat_names,
        "X_full": X_np,
        "y_full": y,
        "buy_dates_full": buy_dates,
        "pnl_test": y[split:],
    }

    if len(X_test) > 0:
        pred = model.predict(X_test)
        ic, pval = spearmanr(y_test, pred)
        ss_res = float(np.sum((y_test - pred) ** 2))
        ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        metrics["ic"] = float(ic)
        metrics["ic_pval"] = float(pval)
        metrics["r2"] = float(r2)
    else:
        metrics["ic"] = metrics["ic_pval"] = metrics["r2"] = float("nan")

    metrics["shap_explainer"], metrics["shap_values"], metrics["X_test_np"] = (
        _compute_shap(model, X_train.to_numpy(), X_test_np)
    )

    return metrics


# ---------------------------------------------------------------------------
# 公共辅助
# ---------------------------------------------------------------------------


def _prepare_xy(trades_feat: pl.DataFrame, feature_col_names: list[str]):
    """提取特征矩阵 X 和目标 y，按 buy_date 排序（时间序列，不打乱）。"""
    import numpy as np
    import pandas as pd

    # 只选 DataFrame 中实际存在的特征列（某些策略数据可能缺少部分列）
    actual_feat_cols = [c for c in feature_col_names if c in trades_feat.columns]
    df_sorted = trades_feat.sort("buy_date")
    # pandas DataFrame 保留列名，供 LightGBM 训练使用（避免 feature names warning）
    X_df = pd.DataFrame(
        df_sorted.select(actual_feat_cols).to_numpy().astype(np.float64),
        columns=actual_feat_cols,
    )
    y = df_sorted["pnl_ret"].to_numpy().astype(np.float64)
    buy_dates = df_sorted["buy_date"].to_numpy()
    return X_df, y, actual_feat_cols, buy_dates


def _compute_shap(model, X_train, X_test):
    """计算 SHAP 值；X_test 空时返回 None。"""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        if len(X_test) > 0:
            shap_vals = explainer.shap_values(X_test)
            # 分类器返回 list[array]，取 class-1 的 shap
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            return explainer, shap_vals, X_test
        return explainer, None, None
    except Exception:  # noqa: BLE001
        return None, None, None


def _save_df(df: pl.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.write_parquet(path)
    else:
        path = (
            path.with_suffix(".csv") if path.suffix.lower() not in (".csv",) else path
        )
        df.write_csv(path)
    console.print(f"[bold green]💾 交易明细已保存[/bold green] → {path}")


# ---------------------------------------------------------------------------
# 终端摘要
# ---------------------------------------------------------------------------


def _print_analysis_summary(
    trades_feat: pl.DataFrame,
    bad_threshold: float,
    filter_result: dict,
    rank_result: dict | None,
) -> None:
    pnl = trades_feat["pnl_ret"]
    n = len(pnl)
    n_bad = int((pnl <= bad_threshold).sum())
    win_rate = float((pnl > 0).sum()) / n if n > 0 else float("nan")

    table = Table(title="交易统计", show_header=True, header_style="bold magenta")
    table.add_column("指标", style="cyan", min_width=22)
    table.add_column("数值", justify="right", min_width=12)

    table.add_row("交易总数", str(n))
    table.add_row("坏交易数（≤ threshold）", str(n_bad))
    table.add_row("胜率（pnl > 0）", f"{win_rate:.2%}")
    avg_pnl = float(pnl.mean() or 0.0)
    table.add_row("平均收益", f"{avg_pnl:.4f}")
    avg_hold = float(trades_feat["hold_days"].mean() or 0.0)
    table.add_row("平均持仓天数", f"{avg_hold:.1f}")

    table.add_row("[bold]── filter 分析 ──[/bold]", "")
    table.add_row(
        "训练/测试样本", f"{filter_result['n_train']} / {filter_result['n_test']}"
    )
    table.add_row("AUC（坏交易=1）", f"{filter_result.get('auc', float('nan')):.4f}")
    table.add_row("Precision", f"{filter_result.get('precision', float('nan')):.4f}")
    table.add_row("Recall", f"{filter_result.get('recall', float('nan')):.4f}")

    if rank_result is not None:
        table.add_row("[bold]── rank 分析 ──[/bold]", "")
        table.add_row(
            "训练/测试样本", f"{rank_result['n_train']} / {rank_result['n_test']}"
        )
        table.add_row("Spearman IC", f"{rank_result.get('ic', float('nan')):.4f}")
        table.add_row("R²", f"{rank_result.get('r2', float('nan')):.4f}")

    console.print(table)
