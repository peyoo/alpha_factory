"""tests/test_cli_pure.py — `quant pure` 命令单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
from typer.testing import CliRunner

from alpha_factory.cli.main import app

runner = CliRunner()

# ── 辅助：mock 数据 ───────────────────────────────────────────────────────────

_MOCK_DAILY = pl.DataFrame(
    {
        "DATE": ["2022-01-04", "2022-01-05"],
        "NAV": [1.0, 1.02],
        "NET_RET": [0.0, 0.02],
        "TURNOVER": [0.05, 0.04],
        "COUNT": [10, 10],
    }
)

_MOCK_TRADES = pl.DataFrame(
    {
        "asset": ["000001.SZ", "000002.SZ"],
        "entry_date": ["2022-01-04", "2022-01-04"],
        "exit_date": ["2022-01-06", "2022-01-07"],
        "pnl_ret": [0.01, -0.005],
        "holding_periods": [2, 3],
    }
)

_MOCK_BT_RESULT = {
    "daily_results": _MOCK_DAILY,
    "trade_details": _MOCK_TRADES,
}

_MOCK_DF = pl.DataFrame(
    {
        "DATE": ["2022-01-04", "2022-01-05"],
        "ASSET": ["000001.SZ", "000002.SZ"],
        "POOL_MASK": [True, True],
        "rank_f1": [0.5, -0.3],
        "pure_f1": [0.8, 0.2],
        "CLOSE": [10.0, 11.0],
        "VWAP": [10.1, 10.9],
        "IS_SUSPENDED": [False, False],
        "IS_UP_LIMIT": [False, False],
        "IS_DOWN_LIMIT": [False, False],
    }
)


def _make_mock_cfg(ranks_empty: bool = False) -> MagicMock:
    """构造一个模拟的 StrategyConfig 实例。"""
    mock_cfg = MagicMock()
    mock_cfg.pool = "main_small_pool"
    mock_cfg.start_date = "20200101"
    mock_cfg.end_date = None
    mock_cfg.hold_num = 10
    mock_cfg.sell_rank = 30
    mock_cfg.cost = 0.003
    mock_cfg.exe_price = "vwap"
    mock_cfg.n_bins = 10

    if ranks_empty:
        mock_cfg.ranks = []
        mock_cfg.ranked_factor_exprs = []
    else:
        mock_rank = MagicMock()
        mock_rank.name = "rank_f1"
        mock_rank.expr_str = "rank_f1 = ts_mean(AMOUNT, 40)"
        mock_cfg.ranks = [mock_rank]
        mock_cfg.ranked_factor_exprs = ["rank_f1 = ts_mean(AMOUNT, 40)"]

    mock_cfg.get_condition_exprs.return_value = []
    return mock_cfg


def _patch_all(ranks_empty: bool = False):
    """返回用于 `quant pure` 的标准 patch 上下文组合。"""
    mock_cfg = _make_mock_cfg(ranks_empty=ranks_empty)
    mock_lf = MagicMock()
    mock_lf.collect.return_value = _MOCK_DF

    ctx_cfg = patch(
        "alpha_factory.cli.pure.StrategyConfig.from_yaml",
        return_value=mock_cfg,
    )
    ctx_dp = patch(
        "alpha_factory.cli.pure.DataProvider",
        return_value=MagicMock(load_pool_data=MagicMock(return_value=mock_lf)),
    )
    ctx_pool = patch(
        "alpha_factory.cli._loader.resolve_pool",
        return_value=MagicMock(),
    )
    ctx_pure = patch(
        "alpha_factory.cli.pure._compute_pure_factor",
        return_value=_MOCK_DF,
    )
    ctx_bt = patch(
        "alpha_factory.cli.pure.backtest_quick_daily",
        return_value=_MOCK_BT_RESULT,
    )
    ctx_report = patch("alpha_factory.cli.pure.generate_and_open_report")
    ctx_print = patch("alpha_factory.cli.backtest._print_summary")
    return ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt, ctx_report, ctx_print, mock_cfg


# ── 测试：基本流程 ────────────────────────────────────────────────────────────


def _make_dummy_yaml(tmp_path: Path) -> Path:
    """创建一个最小化的 YAML 文件以通过 CLI 的文件存在性检查。"""
    yaml_path = tmp_path / "s1.yaml"
    yaml_path.write_text("pool: main_small_pool\npool_config: {}\n")
    return yaml_path


def test_pure_basic(tmp_path: Path):
    """正常调用，exit_code=0，且回测函数被调用。"""
    yaml_file = _make_dummy_yaml(tmp_path)
    ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt, ctx_report, ctx_print, _ = _patch_all()
    with ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt as mock_bt, ctx_report, ctx_print:
        result = runner.invoke(
            app,
            [
                "pure",
                "-y",
                str(yaml_file),
                "--expr",
                "pure_f1=CLOSE/OPEN",
                "--no-report",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_bt.assert_called_once()
    call_kwargs = mock_bt.call_args.kwargs
    assert call_kwargs["factor_col"] == "pure_f1"


def test_pure_expr_without_name_defaults_to_pure_f1(tmp_path: Path):
    """无 `=` 时因子名默认为 `pure_f1`。"""
    yaml_file = _make_dummy_yaml(tmp_path)
    ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt, ctx_report, ctx_print, _ = _patch_all()
    with ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt as mock_bt, ctx_report, ctx_print:
        result = runner.invoke(
            app,
            ["pure", "-y", str(yaml_file), "--expr", "CLOSE/OPEN", "--no-report"],
        )
    assert result.exit_code == 0, result.output
    call_kwargs = mock_bt.call_args.kwargs
    assert call_kwargs["factor_col"] == "pure_f1"


def test_pure_uses_config_start_date(tmp_path: Path):
    """无 `-s` 参数时使用默认值 '20190101'。"""
    yaml_file = _make_dummy_yaml(tmp_path)
    ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt, ctx_report, ctx_print, _ = _patch_all()
    with (
        ctx_cfg,
        ctx_dp as mock_dp_cls,
        ctx_pool,
        ctx_pure,
        ctx_bt,
        ctx_report,
        ctx_print,
    ):
        result = runner.invoke(
            app,
            ["pure", "-y", str(yaml_file), "--expr", "CLOSE/OPEN", "--no-report"],
        )
    assert result.exit_code == 0, result.output
    # 默认 start_date 为 "20190101"（CLI 参数默认值）
    call_args = mock_dp_cls.return_value.load_pool_data.call_args
    assert call_args.args[1] == "20190101"


def test_pure_empty_ranks_error(tmp_path: Path):
    """`cfg.ranks` 为空时，命令应报错并以非零码退出。"""
    yaml_file = _make_dummy_yaml(tmp_path)
    ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt, ctx_report, ctx_print, _ = _patch_all(
        ranks_empty=True
    )
    with ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt, ctx_report, ctx_print:
        result = runner.invoke(
            app,
            ["pure", "-y", str(yaml_file), "--expr", "CLOSE/OPEN", "--no-report"],
        )
    assert result.exit_code != 0


def test_pure_save_trades(tmp_path: Path):
    """`--save-trades` 时，回测结果被写入文件。"""
    yaml_file = _make_dummy_yaml(tmp_path)
    out_file = tmp_path / "trades.csv"
    mock_trades = MagicMock()

    mock_bt_result = {
        "daily_results": _MOCK_DAILY,
        "trade_details": mock_trades,
    }

    ctx_cfg, ctx_dp, ctx_pool, ctx_pure, _, ctx_report, ctx_print, _ = _patch_all()
    with (
        ctx_cfg,
        ctx_dp,
        ctx_pool,
        ctx_pure,
        patch(
            "alpha_factory.cli.pure.backtest_quick_daily", return_value=mock_bt_result
        ),
        ctx_report,
        ctx_print,
    ):
        result = runner.invoke(
            app,
            [
                "pure",
                "-y",
                str(yaml_file),
                "--expr",
                "CLOSE/OPEN",
                "--no-report",
                "--save-trades",
                str(out_file),
            ],
        )
    assert result.exit_code == 0, result.output
    mock_trades.write_csv.assert_called_once()


def test_pure_all_exprs_include_target_factor(tmp_path: Path):
    """load_pool_data 的 exprs 参数中包含目标因子表达式。"""
    yaml_file = _make_dummy_yaml(tmp_path)
    ctx_cfg, ctx_dp, ctx_pool, ctx_pure, ctx_bt, ctx_report, ctx_print, _ = _patch_all()
    with (
        ctx_cfg,
        ctx_dp as mock_dp_cls,
        ctx_pool,
        ctx_pure,
        ctx_bt,
        ctx_report,
        ctx_print,
    ):
        runner.invoke(
            app,
            [
                "pure",
                "-y",
                str(yaml_file),
                "--expr",
                "my_f=ts_mean(AMOUNT,40)",
                "--no-report",
            ],
        )
    call_kwargs = mock_dp_cls.return_value.load_pool_data.call_args.kwargs
    exprs = call_kwargs.get("exprs", [])
    assert any("my_f" in e for e in exprs), f"期望 exprs 含目标因子，实际: {exprs}"
