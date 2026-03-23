from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import yaml
from typer.testing import CliRunner

from alpha_factory.cli.main import app
from alpha_factory.utils.schema import F

runner = CliRunner()

# ---------------------------------------------------------------------------
# 共享 Fixtures
# ---------------------------------------------------------------------------

_SINGLE_RANK_YAML = {
    "name": "test_single",
    "pool": "main_small_pool",
    "hold_num": 10,
    "sell_rank": 30,
    "cost": 0.003,
    "exe_price": "vwap",
    "ranks": [
        {
            "name": "f1",
            "expression": "-ts_mean(AMOUNT, 60)",
            "weight": 1.0,
            "direction": -1,
        }
    ],
}

_MULTI_RANK_YAML = {
    "name": "test_multi",
    "pool": "main_small_pool",
    "hold_num": 10,
    "sell_rank": 30,
    "cost": 0.003,
    "exe_price": "vwap",
    "ranks": [
        {
            "name": "f1",
            "expression": "-ts_mean(AMOUNT, 60)",
            "weight": 0.5,
            "direction": -1,
        },
        {"name": "f2", "expression": "CLOSE / VWAP", "weight": 0.5, "direction": 1},
    ],
}


def _write_yaml(tmp_path: Path, data: dict, name: str = "strategy.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
    return p


def _build_mock_lf() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            F.DATE: ["2026-01-02", "2026-01-02"],
            F.ASSET: ["AAA", "BBB"],
            F.POOL_MASK: [True, True],
            "f1": [1.0, 3.0],
            F.VWAP: [10.0, 11.0],
            F.CLOSE: [10.1, 11.2],
            F.IS_UP_LIMIT: [False, False],
            F.IS_DOWN_LIMIT: [False, False],
            F.IS_SUSPENDED: [False, False],
        }
    ).lazy()


def _fake_backtest(**kwargs):
    return {
        "daily_results": pl.DataFrame(
            {
                F.DATE: ["2026-01-02"],
                "NET_RET": [0.0],
                "TURNOVER": [0.0],
                "COUNT": [0],
                "NAV": [1.0],
            }
        ),
        "trade_details": pl.DataFrame([]),
    }


# ---------------------------------------------------------------------------
# 单因子：直接加载原始表达式
# ---------------------------------------------------------------------------


def test_bt_single_factor_loads_raw_expr(tmp_path: Path):
    """单因子时应直接将 expression 传给 DataProvider，不经过 rank 预计算。"""
    yaml_path = _write_yaml(tmp_path, _SINGLE_RANK_YAML)

    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ),
        patch("alpha_factory.cli.backtest._print_summary"),
        patch("alpha_factory.cli.opt._resolve_pool"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(app, ["bt", "-y", str(yaml_path), "--no-report"])

    assert result.exit_code == 0, result.output
    call_exprs = mock_dp_cls.return_value.load_pool_data.call_args.kwargs["exprs"]
    # 因子名字在 from_yaml 时统一生成为 rank_f1
    assert call_exprs == ["rank_f1 = -ts_mean(AMOUNT, 60)"]


def test_bt_single_factor_direction_minus1_ascending_false(tmp_path: Path):
    """方向已在表达式生成时统一处理，backtest 层级统一使用 ascending=False。"""
    yaml_path = _write_yaml(tmp_path, _SINGLE_RANK_YAML)

    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ) as mock_bt,
        patch("alpha_factory.cli.backtest._print_summary"),
        patch("alpha_factory.cli.opt._resolve_pool"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(app, ["bt", "-y", str(yaml_path), "--no-report"])

    assert result.exit_code == 0, result.output
    assert mock_bt.call_args.kwargs["ascending"] is False


def test_bt_single_factor_direction_1_ascending_false(tmp_path: Path):
    """单因子时 backtest 层级统一使用 ascending=False。"""
    data = {
        **_SINGLE_RANK_YAML,
        "ranks": [{"name": "f1", "expression": "CLOSE", "weight": 1.0, "direction": 1}],
    }
    yaml_path = _write_yaml(tmp_path, data)

    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ) as mock_bt,
        patch("alpha_factory.cli.backtest._print_summary"),
        patch("alpha_factory.cli.opt._resolve_pool"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(app, ["bt", "-y", str(yaml_path), "--no-report"])

    assert result.exit_code == 0, result.output
    assert mock_bt.call_args.kwargs["ascending"] is False


# ---------------------------------------------------------------------------
# 多因子：load_pool_data + Rank.process 合成 COMPOSITE_OPT
# ---------------------------------------------------------------------------


def test_bt_multi_factor_uses_composite(tmp_path: Path):
    """多因子时应通过 Rank.process 合成 COMPOSITE_OPT，backtest ascending=False。"""
    yaml_path = _write_yaml(tmp_path, _MULTI_RANK_YAML)

    mock_base_df = pl.DataFrame(
        {
            F.DATE: ["2026-01-02"],
            F.ASSET: ["AAA"],
            F.POOL_MASK: [True],
            F.VWAP: [10.0],
            F.CLOSE: [10.1],
            F.IS_UP_LIMIT: [False],
            F.IS_DOWN_LIMIT: [False],
            F.IS_SUSPENDED: [False],
            "f1": [1.0],
            "f2": [2.0],
            "COMPOSITE_OPT": [1.5],
        }
    )

    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ) as mock_bt,
        patch("alpha_factory.cli.backtest._print_summary"),
        patch("alpha_factory.cli.opt._resolve_pool"),
        patch(
            "alpha_factory.data_provider.factorsprocessor.FactorsRankComposite"
        ) as mock_rank_composite,
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = mock_base_df.lazy()
        mock_rank_composite.return_value.process.return_value = mock_base_df
        result = runner.invoke(app, ["bt", "-y", str(yaml_path), "--no-report"])

    assert result.exit_code == 0, result.output
    assert mock_rank_composite.called
    assert mock_bt.call_args.kwargs["ascending"] is False


# ---------------------------------------------------------------------------
# 参数透传：hold_num / sell_rank / cost 来自 YAML
# ---------------------------------------------------------------------------


def test_bt_backtest_params_from_yaml(tmp_path: Path):
    """hold_num / sell_rank / cost 应从 YAML 透传到 backtest_daily_evolving。"""
    data = {**_SINGLE_RANK_YAML, "hold_num": 25, "sell_rank": 50, "cost": 0.001}
    yaml_path = _write_yaml(tmp_path, data)

    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ) as mock_bt,
        patch("alpha_factory.cli.backtest._print_summary"),
        patch("alpha_factory.cli.opt._resolve_pool"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(app, ["bt", "-y", str(yaml_path), "--no-report"])

    assert result.exit_code == 0, result.output
    kw = mock_bt.call_args.kwargs
    assert kw["n_buy"] == 25
    assert kw["sell_rank"] == 50
    assert kw["cost_rate"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


def test_bt_yaml_file_not_found_exits():
    result = runner.invoke(app, ["bt", "-y", "/nonexistent/s.yaml", "--no-report"])
    assert result.exit_code != 0


def test_bt_yaml_empty_ranks_exits(tmp_path: Path):
    data = {**_SINGLE_RANK_YAML, "ranks": []}
    yaml_path = _write_yaml(tmp_path, data)

    with patch("alpha_factory.cli.opt._resolve_pool"):
        result = runner.invoke(app, ["bt", "-y", str(yaml_path), "--no-report"])

    assert result.exit_code != 0


def test_bt_yaml_required():
    """不传 --yaml 时 typer 应报错退出。"""
    result = runner.invoke(app, ["bt", "--no-report"])
    assert result.exit_code != 0
