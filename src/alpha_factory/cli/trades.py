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
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from alpha_factory.cli._loader import resolve_pool
from alpha_factory.cli._trades_viz import (
    _shap_to_base64,
    _detect_shap_threshold,
    _shap_dependence_grid,
    _shap_heatmap_by_time,
    _shap_interaction_plots,
    _shap_decision_plots,
)
from alpha_factory.cli.opt import _COMPOSITE_COL
from alpha_factory.cli.utils import resolve_yaml_path
from alpha_factory.config.base import settings
from alpha_factory.config.strategy import StrategyConfig
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.utils.schema import F

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
        .fill_null(999999)
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


# ---------------------------------------------------------------------------
# HTML 报告
# ---------------------------------------------------------------------------


def _build_diagnosis_html(
    filter_result: dict,
    rank_result: dict | None,
    bad_threshold: float,
    mode: str = "both",
) -> str:
    """根据 SHAP 均值自动生成中文诊断卡片 HTML。"""
    try:
        import numpy as np

        f_shap = filter_result.get("shap_values")
        r_shap = rank_result.get("shap_values") if rank_result is not None else None
        f_names = filter_result.get("feature_names", [])
        r_names = (
            rank_result.get("feature_names", []) if rank_result is not None else []
        )

        if f_shap is None:
            return ""

        f_mean = np.abs(f_shap).mean(axis=0)

        # 坏交易最危险因子：filter top-3
        f_top_idx = np.argsort(f_mean)[::-1][:3]
        f_top = [(f_names[i], float(f_mean[i])) for i in f_top_idx if i < len(f_names)]

        # 双确认（同时出现在 filter top-5 和 rank top-5）
        f_top5_names = {
            f_names[i] for i in np.argsort(f_mean)[::-1][:5] if i < len(f_names)
        }
        r_top: list[tuple[str, float]] = []
        dual: list[str] = []
        if r_shap is not None:
            r_mean = np.abs(r_shap).mean(axis=0)
            # 高收益核心因子：rank top-3
            r_top_idx = np.argsort(r_mean)[::-1][:3]
            r_top = [
                (r_names[i], float(r_mean[i])) for i in r_top_idx if i < len(r_names)
            ]
            r_top5_names = {
                r_names[i] for i in np.argsort(r_mean)[::-1][:5] if i < len(r_names)
            }
            dual = sorted(f_top5_names & r_top5_names)

        def card(icon, title, body, color):
            return f"""<div style="background:{color};border-radius:8px;padding:12px 16px;margin:8px 0;border-left:4px solid #1a5276">
<b style="font-size:1.05em">{icon} {title}</b>
<div style="margin-top:6px;line-height:1.7;font-size:.92em">{body}</div>
</div>"""

        # 卡片1：防御
        shield_rows = "".join(
            f"<li><b>{n}</b> (|SHAP| 均值={v:.4f})"
            + (
                lambda t: (
                    f"，非线性阈值 ≈ <b>{t:.3g}</b>，建议在该阈值处过滤"
                    if t is not None
                    else "，暂未检测到明显非线性阈值"
                )
            )(
                _detect_shap_threshold(
                    filter_result.get("X_full", np.zeros((1, len(f_names))))[
                        :, f_names.index(n)
                    ]
                    if n in f_names
                    else np.array([0]),
                    f_shap[:, f_names.index(n)] if n in f_names else np.array([0]),
                )
                if f_names
                else None
            )
            + "</li>"
            for n, v in f_top
        )
        card1 = card(
            "🛡",
            "防御盾牌 — 坏交易风险因子",
            f"以下因子对 <b>pnl_ret ≤ {bad_threshold}</b> 的坏交易贡献最大，建议加入 <code>pool_mask</code> / <code>not_buy_able</code>：<ul>{shield_rows}</ul>",
            "#fff3f3",
        )

        # 卡片2：进攻
        atk_rows = "".join(
            f"<li><b>{n}</b> (|SHAP| 均值={v:.4f})</li>" for n, v in r_top
        )
        card2 = card(
            "⚔",
            "进攻矛头 — 高收益驱动因子",
            f"以下因子对高收益 <b>pnl_ret</b> 贡献最大，建议加入/增权 <code>ranks</code>：<ul>{atk_rows}</ul>",
            "#f3fff3",
        )

        # 卡片3：双确认
        if dual:
            dual_items = "".join(f"<li><b>{n}</b></li>" for n in dual)
            card3 = card(
                "🔥",
                "双确认因子 — 排雷选优均显著",
                f"以下因子同时出现在 filter top-5 和 rank top-5，表明其对交易结果有强烈的非线性影响，建议重点审查：<ul>{dual_items}</ul>",
                "#fffbf0",
            )
        else:
            card3 = card(
                "🔥",
                "双确认因子",
                "未检测到同时出现在 filter/rank top-5 的因子，各维度独立发挥作用。",
                "#fffbf0",
            )

        cards_html = "\n".join(
            [
                *((card1,) if mode in ("filter", "both") else ()),
                *((card2,) if mode in ("rank", "both") else ()),
                card3,
            ]
        )
        return f"""<div class="section" id="diagnosis">
<h2>🩺 自动诊断卡片</h2>
<p class="hint">以下诊断由 SHAP 值自动生成，供策略调优参考。</p>
{cards_html}
</div>"""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# HTML 主报告生成
# ---------------------------------------------------------------------------


def _generate_html_report(
    cfg: StrategyConfig,
    trades_feat: pl.DataFrame,
    bad_threshold: float,
    filter_result: dict,
    rank_result: dict | None,
    full_diag: bool = False,
) -> tuple[Path, Path | None]:
    import numpy as np

    f_names = filter_result.get("feature_names", [])
    r_names = rank_result.get("feature_names", []) if rank_result is not None else []

    # ── Section 1：Global Landscape（beeswarm + bar）──────────────────────
    console.print("[dim]  [1/5] Global landscape (beeswarm + bar)...[/dim]")
    f_bee = _shap_to_base64(
        filter_result["shap_values"],
        filter_result["X_test_np"],
        f_names,
        "beeswarm",
        "filter: SHAP beeswarm",
    )
    f_bar = _shap_to_base64(
        filter_result["shap_values"],
        filter_result["X_test_np"],
        f_names,
        "bar",
        "filter: Feature Importance",
    )
    r_bee = (
        _shap_to_base64(
            rank_result["shap_values"],
            rank_result["X_test_np"],
            r_names,
            "beeswarm",
            "rank: SHAP beeswarm",
        )
        if rank_result is not None
        else ""
    )
    r_bar = (
        _shap_to_base64(
            rank_result["shap_values"],
            rank_result["X_test_np"],
            r_names,
            "bar",
            "rank: Feature Importance",
        )
        if rank_result is not None
        else ""
    )

    # ── Section 2：Dependence Grid ────────────────────────────────────────
    console.print("[dim]  [2/5] Dependence plots...[/dim]")
    f_dep_imgs: list[str] = []
    r_dep_imgs: list[str] = []
    if filter_result.get("shap_values") is not None:
        f_dep_imgs = _shap_dependence_grid(
            filter_result["shap_values"], filter_result["X_test_np"], f_names
        )
    if rank_result is not None and rank_result.get("shap_values") is not None:
        r_dep_imgs = _shap_dependence_grid(
            rank_result["shap_values"], rank_result["X_test_np"], r_names
        )

    # ── Section 3：Interaction（仅 full_diag）────────────────────────────
    f_inter_img = ""
    r_inter_img = ""
    if full_diag:
        console.print("[dim]  [3/5] SHAP Interaction（最慢，请稍候...）[/dim]")
        X_full_f = filter_result.get("X_full")
        if X_full_f is not None and filter_result.get("model") is not None:
            console.print("[dim]        → filter 模型...[/dim]")
            f_inter_img = _shap_interaction_plots(
                filter_result["model"], X_full_f, f_names
            )
        if rank_result is not None:
            X_full_r = rank_result.get("X_full")
            if X_full_r is not None and rank_result.get("model") is not None:
                console.print("[dim]        → rank 模型...[/dim]")
                r_inter_img = _shap_interaction_plots(
                    rank_result["model"], X_full_r, r_names
                )
    else:
        console.print("[dim]  [3/5] Interaction 已跳过（使用 --full-diag 开启）[/dim]")

    # ── Section 4：Time Heatmap ───────────────────────────────────────────
    console.print("[dim]  [4/5] Time heatmap...[/dim]")
    f_heat_img = ""
    r_heat_img = ""
    if (
        filter_result.get("shap_values") is not None
        and filter_result.get("buy_dates_full") is not None
    ):
        f_heat_img = _shap_heatmap_by_time(
            filter_result["shap_values"],
            filter_result["X_test_np"],
            f_names,
            filter_result["buy_dates_full"][
                len(filter_result["buy_dates_full"])
                - len(filter_result["shap_values"]) :
            ],
        )
    if (
        rank_result is not None
        and rank_result.get("shap_values") is not None
        and rank_result.get("buy_dates_full") is not None
    ):
        r_heat_img = _shap_heatmap_by_time(
            rank_result["shap_values"],
            rank_result["X_test_np"],
            r_names,
            rank_result["buy_dates_full"][
                len(rank_result["buy_dates_full"]) - len(rank_result["shap_values"]) :
            ],
        )

    # ── Section 5：Decision Plot（worst-20）──────────────────────────────
    console.print("[dim]  [5/5] Decision plots...[/dim]")
    f_dec_img = ""
    r_dec_img = ""
    worst_n = 20
    if (
        filter_result.get("shap_values") is not None
        and filter_result.get("pnl_test") is not None
        and filter_result.get("shap_explainer") is not None
    ):
        pnl_t = filter_result["pnl_test"]
        sv = filter_result["shap_values"]
        xt = filter_result["X_test_np"]
        if len(pnl_t) >= worst_n:
            worst_idx = np.argsort(pnl_t)[:worst_n]
        else:
            worst_idx = np.argsort(pnl_t)
        f_dec_img = _shap_decision_plots(
            filter_result["shap_explainer"],
            sv[worst_idx],
            xt[worst_idx],
            f_names,
        )
    if (
        rank_result is not None
        and rank_result.get("shap_values") is not None
        and rank_result.get("pnl_test") is not None
        and rank_result.get("shap_explainer") is not None
    ):
        pnl_t = rank_result["pnl_test"]
        sv = rank_result["shap_values"]
        xt = rank_result["X_test_np"]
        if len(pnl_t) >= worst_n:
            worst_idx = np.argsort(pnl_t)[:worst_n]
        else:
            worst_idx = np.argsort(pnl_t)
        r_dec_img = _shap_decision_plots(
            rank_result["shap_explainer"],
            sv[worst_idx],
            xt[worst_idx],
            r_names,
        )

    # ── 公共辅助 ─────────────────────────────────────────────────────────
    def img_tag(b64: str, alt: str) -> str:
        if not b64:
            return f"<p style='color:gray'>[{alt} 图表生成失败]</p>"
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0" alt="{alt}"/>'

    pnl = trades_feat["pnl_ret"]
    n = len(pnl)
    n_bad = int((pnl <= bad_threshold).sum())
    win_rate = float((pnl > 0).sum()) / n if n > 0 else float("nan")

    def fmt(v, fmt_str=".4f"):
        return f"{v:{fmt_str}}" if v == v else "N/A"

    def dep_section(imgs: list[str]) -> str:
        if not imgs:
            return ""
        inner = "\n".join(
            f'<div style="display:inline-block;margin:4px">{img_tag(b, f"dep {i}")}</div>'
            for i, b in enumerate(imgs)
        )
        return (
            '<div style="margin-top:12px">'
            "<h3>Dependence Plots</h3>"
            '<p class="hint">散点 X=特征值，Y=SHAP 值，颜色=最高交互特征。红虚线为非线性阈值（若检测到）。</p>'
            f'<div style="display:flex;flex-wrap:wrap">{inner}</div>'
            "</div>"
        )

    _CSS = (
        "  body {font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:20px;color:#222}\n"
        "  h1 {color:#1a5276;border-bottom:2px solid #1a5276;padding-bottom:8px}\n"
        "  h2 {color:#1f618d;margin-top:32px}\n"
        "  h3 {color:#333;margin-top:16px}\n"
        "  table {border-collapse:collapse;width:100%;margin:12px 0}\n"
        "  th,td {border:1px solid #ddd;padding:8px 12px;text-align:left}\n"
        "  th {background:#2980b9;color:#fff}\n"
        "  tr:nth-child(even) {background:#f5f5f5}\n"
        "  .section {background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:16px;margin:16px 0}\n"
        "  .badge-filter {background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-size:.85em}\n"
        "  .badge-rank {background:#27ae60;color:#fff;padding:2px 8px;border-radius:4px;font-size:.85em}\n"
        "  .hint {color:#555;font-size:.9em;line-height:1.6}\n"
    )

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_meta = (
        f"buy_rank={cfg.buy_rank} &nbsp;|&nbsp; sell_rank={cfg.sell_rank} &nbsp;|&nbsp; "
        f"坏交易阈值={bad_threshold}"
        + ("&nbsp;|&nbsp;<b>full-diag 模式</b>" if full_diag else "")
    )
    stats_section_html = (
        f'<div class="section"><h2>交易统计</h2>'
        f"<table><tr><th>指标</th><th>数值</th></tr>"
        f"<tr><td>交易总数</td><td>{n}</td></tr>"
        f"<tr><td>坏交易数（pnl_ret ≤ {bad_threshold}）</td><td>{n_bad}</td></tr>"
        f"<tr><td>胜率（pnl_ret &gt; 0）</td><td>{win_rate:.2%}</td></tr>"
        f"<tr><td>平均收益</td><td>{fmt(float(pnl.mean() or 0.0))}</td></tr>"
        f"<tr><td>平均持仓天数</td><td>"
        f"{fmt(float(trades_feat['hold_days'].mean() or 0.0), '.1f')}</td></tr>"
        "</table></div>"
    )

    console.print("[dim]  生成诊断卡片...[/dim]")

    # ── Filter 报告 ───────────────────────────────────────────────────────
    f_diag_html = _build_diagnosis_html(
        filter_result, rank_result, bad_threshold, mode="filter"
    )
    f_inter_section = (
        f'<div class="section"><h2>⚡ Section 3 — 交互与协同效应'
        f' <small style="font-size:.7em;color:#888">（--full-diag）</small></h2>'
        f'<p class="hint">SHAP Interaction Value 热图（左）+ 最强交互对散点图（右）。</p>'
        f"{img_tag(f_inter_img, 'filter interaction')}</div>"
        if full_diag
        else ""
    )
    filter_html = (
        f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"/>'
        f"<title>Filter 分析 — 排雷 · {cfg.name}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>🛡 Filter 分析 — 排雷 · {cfg.name}</h1>"
        f'<p style="color:#666">生成时间：{gen_time} &nbsp;|&nbsp; {base_meta}</p>'
        f'<p class="hint">目标：识别导致 <b>pnl_ret ≤ {bad_threshold}</b> 的坏交易特征，'
        f"建议将高风险条件加入 <code>pool_mask</code> / <code>not_buy_able</code>。</p>"
        f"{f_diag_html}{stats_section_html}"
        f'<div class="section"><h2>📊 Section 1 — Global Landscape</h2>'
        f'<p class="hint">LGBMClassifier：坏交易 label=1，AUC 越高则判别力越强。'
        f"正 SHAP → 推高坏交易概率 → 建议过滤。</p>"
        f"<table><tr><th>指标</th><th>数值</th></tr>"
        f"<tr><td>训练样本</td><td>{filter_result['n_train']}</td></tr>"
        f"<tr><td>测试样本</td><td>{filter_result['n_test']}</td></tr>"
        f"<tr><td>AUC</td><td>{fmt(filter_result.get('auc', float('nan')))}</td></tr>"
        f"<tr><td>Precision（坏=1）</td><td>{fmt(filter_result.get('precision', float('nan')))}</td></tr>"
        f"<tr><td>Recall（坏=1）</td><td>{fmt(filter_result.get('recall', float('nan')))}</td></tr>"
        f"</table>"
        f"<h4>SHAP Beeswarm</h4>{img_tag(f_bee, 'filter beeswarm')}"
        f"<h4>Feature Importance</h4>{img_tag(f_bar, 'filter importance')}</div>"
        f'<div class="section"><h2>🔍 Section 2 — 非线性边界检测（Dependence Plots）</h2>'
        f'<p class="hint">散点 X=特征值，Y=SHAP 值，颜色=最高交互特征。红虚线为非线性阈值（若检测到）。</p>'
        f"{dep_section(f_dep_imgs)}</div>"
        f"{f_inter_section}"
        f'<div class="section"><h2>🌡 Section 4 — 稳定性与风格漂移（SHAP 时间热图）</h2>'
        f'<p class="hint">行=按时间排序的交易，列=特征，颜色=SHAP 值（红=正贡献，蓝=负贡献）。</p>'
        f"{img_tag(f_heat_img, 'filter heatmap')}</div>"
        f'<div class="section"><h2>🏚 Section 5 — 诊断调试（最差 {worst_n} 笔交易 Decision Plot）</h2>'
        f'<p class="hint">Decision Plot 展示各特征如何将模型预测值从 baseline 推向最终输出。</p>'
        f"{img_tag(f_dec_img, 'filter decision')}</div>"
        "</body></html>"
    )

    # ── Rank 报告 ─────────────────────────────────────────────────────────
    r_diag_html = _build_diagnosis_html(
        filter_result, rank_result, bad_threshold, mode="rank"
    )
    r_inter_section = (
        f'<div class="section"><h2>⚡ Section 3 — 交互与协同效应'
        f' <small style="font-size:.7em;color:#888">（--full-diag）</small></h2>'
        f'<p class="hint">SHAP Interaction Value 热图（左）+ 最强交互对散点图（右）。</p>'
        f"{img_tag(r_inter_img, 'rank interaction')}</div>"
        if full_diag
        else ""
    )
    rank_html = (
        (
            f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"/>'
            f"<title>Rank 分析 — 选优 · {cfg.name}</title>"
            f"<style>{_CSS}</style></head><body>"
            f"<h1>⚔ Rank 分析 — 选优 · {cfg.name}</h1>"
            f'<p style="color:#666">生成时间：{gen_time} &nbsp;|&nbsp; {base_meta}</p>'
            f'<p class="hint">目标：识别预测高 <b>pnl_ret</b> 的特征，'
            f"建议将高权重因子加入 <code>ranks</code>。</p>"
            f"{r_diag_html}{stats_section_html}"
            f'<div class="section"><h2>📊 Section 1 — Global Landscape</h2>'
            f'<p class="hint">LGBMRegressor：目标 pnl_ret，Spearman IC 越高则预测能力越强。'
            f"正 SHAP → 推高收益 → 建议增权。</p>"
            f"<table><tr><th>指标</th><th>数值</th></tr>"
            f"<tr><td>训练样本</td><td>{rank_result['n_train']}</td></tr>"
            f"<tr><td>测试样本</td><td>{rank_result['n_test']}</td></tr>"
            f"<tr><td>Spearman IC</td><td>{fmt(rank_result.get('ic', float('nan')))}</td></tr>"
            f"<tr><td>IC p-value</td><td>{fmt(rank_result.get('ic_pval', float('nan')))}</td></tr>"
            f"<tr><td>R²</td><td>{fmt(rank_result.get('r2', float('nan')))}</td></tr>"
            f"</table>"
            f"<h4>SHAP Beeswarm</h4>{img_tag(r_bee, 'rank beeswarm')}"
            f"<h4>Feature Importance</h4>{img_tag(r_bar, 'rank importance')}</div>"
            f'<div class="section"><h2>🔍 Section 2 — 非线性边界检测（Dependence Plots）</h2>'
            f'<p class="hint">散点 X=特征值，Y=SHAP 值，颜色=最高交互特征。红虚线为非线性阈值（若检测到）。</p>'
            f"{dep_section(r_dep_imgs)}</div>"
            f"{r_inter_section}"
            f'<div class="section"><h2>🌡 Section 4 — 稳定性与风格漂移（SHAP 时间热图）</h2>'
            f'<p class="hint">行=按时间排序的交易，列=特征，颜色=SHAP 值（红=正贡献，蓝=负贡献）。</p>'
            f"{img_tag(r_heat_img, 'rank heatmap')}</div>"
            f'<div class="section"><h2>🏚 Section 5 — 诊断调试（最差 {worst_n} 笔交易 Decision Plot）</h2>'
            f'<p class="hint">Decision Plot 展示各特征如何将预测值从 baseline 推向最终输出。</p>'
            f"{img_tag(r_dec_img, 'rank decision')}</div>"
            "</body></html>"
        )
        if rank_result is not None
        else None
    )

    report_dir = Path(settings.OUTPUT_DIR) / "html_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_path = report_dir / f"Trades_filter_{cfg.name}_{ts}.html"
    rank_path: Path | None = None
    filter_path.write_text(filter_html, encoding="utf-8")
    if rank_html is not None:
        rank_path = report_dir / f"Trades_rank_{cfg.name}_{ts}.html"
        rank_path.write_text(rank_html, encoding="utf-8")
    return filter_path, rank_path
