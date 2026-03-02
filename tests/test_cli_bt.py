from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import polars as pl
from typer.testing import CliRunner

from alpha_factory.cli.main import app
from alpha_factory.data_provider.pool import MainSmallPool
from alpha_factory.utils.schema import F

runner = CliRunner()


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
                "RAW_RET": [0.0],
                "NET_RET": [0.0],
                "TURNOVER": [0.0],
                "COUNT": [0],
                "NAV": [1.0],
            }
        ),
        "trade_details": pl.DataFrame([]),
    }


def test_bt_default_preprocess_mode_none_keeps_raw_expr():
    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ),
        patch("alpha_factory.cli.backtest._print_summary"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(
            app,
            ["bt", "--expr", "f1=CLOSE", "--no-report"],
        )

    assert result.exit_code == 0, result.output
    assert mock_dp_cls.return_value.load_pool_data.call_args.kwargs["exprs"] == [
        "f1 = CLOSE"
    ]


def test_bt_preprocess_mode_demean_wraps_expr():
    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ),
        patch("alpha_factory.cli.backtest._print_summary"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(
            app,
            ["bt", "--expr", "f1=CLOSE", "--preprocess-mode", "demean", "--no-report"],
        )

    assert result.exit_code == 0, result.output
    assert mock_dp_cls.return_value.load_pool_data.call_args.kwargs["exprs"] == [
        "f1 = cs_demean_mask(CLOSE)"
    ]


def test_bt_preprocess_mode_zscore_wraps_unnamed_expr():
    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ),
        patch("alpha_factory.cli.backtest._print_summary"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(
            app,
            [
                "bt",
                "--expr",
                "-ts_mean(AMOUNT,60)",
                "--preprocess-mode",
                "zscore",
                "--no-report",
            ],
        )

    assert result.exit_code == 0, result.output
    assert mock_dp_cls.return_value.load_pool_data.call_args.kwargs["exprs"] == [
        "f1 = cs_mad_zscore_mask(-ts_mean(AMOUNT,60))"
    ]


def test_bt_invalid_preprocess_mode_exits_with_error():
    with patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls:
        mock_dp_cls.return_value.load_pool_data = MagicMock()
        result = runner.invoke(
            app,
            [
                "bt",
                "--expr",
                "f1=CLOSE",
                "--preprocess-mode",
                "bad_mode",
                "--no-report",
            ],
        )

    assert result.exit_code != 0
    assert "--preprocess-mode" in result.output


def test_bt_csv_builds_equal_weight_expr(tmp_path: Path):
    csv_file = tmp_path / "factors.csv"
    pl.DataFrame(
        {
            "factor": ["a", "b"],
            "expression": ["a = -ts_mean(AMOUNT,60)", "b = cs_rank_mask(CLOSE)"],
        }
    ).write_csv(csv_file)

    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ),
        patch("alpha_factory.cli.backtest._print_summary"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(
            app,
            ["bt", "--csv", str(csv_file), "--no-report"],
        )

    assert result.exit_code == 0, result.output
    assert mock_dp_cls.return_value.load_pool_data.call_args.kwargs["exprs"] == [
        "f_csv = ((-ts_mean(AMOUNT,60)) + (cs_rank_mask(CLOSE))) / 2"
    ]


def test_bt_csv_preprocess_applies_to_each_expr(tmp_path: Path):
    csv_file = tmp_path / "factors.csv"
    pl.DataFrame(
        {
            "expression": ["-ts_mean(AMOUNT,60)", "CLOSE / VWAP"],
        }
    ).write_csv(csv_file)

    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ),
        patch("alpha_factory.cli.backtest._print_summary"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(
            app,
            [
                "bt",
                "--csv",
                str(csv_file),
                "--preprocess-mode",
                "demean",
                "--no-report",
            ],
        )

    assert result.exit_code == 0, result.output
    assert mock_dp_cls.return_value.load_pool_data.call_args.kwargs["exprs"] == [
        "f_csv = ((cs_demean_mask(-ts_mean(AMOUNT,60))) + (cs_demean_mask(CLOSE / VWAP))) / 2"
    ]


def test_bt_csv_direction_controls_plus_minus(tmp_path: Path):
    csv_file = tmp_path / "factors.csv"
    pl.DataFrame(
        {
            "expression": ["-ts_mean(AMOUNT,60)", "CLOSE / VWAP"],
            "direction": [1, -1],
        }
    ).write_csv(csv_file)

    with (
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ),
        patch("alpha_factory.cli.backtest._print_summary"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(
            app,
            ["bt", "--csv", str(csv_file), "--no-report"],
        )

    assert result.exit_code == 0, result.output
    assert mock_dp_cls.return_value.load_pool_data.call_args.kwargs["exprs"] == [
        "f_csv = ((-ts_mean(AMOUNT,60)) + -(CLOSE / VWAP)) / 2"
    ]


def test_bt_csv_relative_path_resolves_from_pool_dir(tmp_path: Path):
    csv_file = tmp_path / "factors.csv"
    pl.DataFrame(
        {
            "expression": ["-ts_mean(AMOUNT,60)", "CLOSE"],
        }
    ).write_csv(csv_file)

    with (
        patch.object(
            MainSmallPool,
            "pool_dir",
            new_callable=PropertyMock,
            return_value=tmp_path,
        ),
        patch("alpha_factory.cli.backtest.DataProvider") as mock_dp_cls,
        patch(
            "alpha_factory.cli.backtest.backtest_daily_evolving",
            side_effect=_fake_backtest,
        ),
        patch("alpha_factory.cli.backtest._print_summary"),
    ):
        mock_dp_cls.return_value.load_pool_data.return_value = _build_mock_lf()
        result = runner.invoke(
            app,
            ["bt", "--csv", "factors.csv", "--no-report"],
        )

    assert result.exit_code == 0, result.output
    assert mock_dp_cls.return_value.load_pool_data.call_args.kwargs["exprs"] == [
        "f_csv = ((-ts_mean(AMOUNT,60)) + (CLOSE)) / 2"
    ]


def test_bt_expr_and_csv_mutually_exclusive(tmp_path: Path):
    csv_file = tmp_path / "factors.csv"
    pl.DataFrame({"expression": ["CLOSE"]}).write_csv(csv_file)

    result = runner.invoke(
        app,
        ["bt", "--expr", "f1=CLOSE", "--csv", str(csv_file), "--no-report"],
    )

    assert result.exit_code != 0
    assert "--expr" in result.output and "--csv" in result.output


def test_bt_requires_expr_or_csv():
    result = runner.invoke(
        app,
        ["bt", "--no-report"],
    )

    assert result.exit_code != 0
    assert "--expr" in result.output and "--csv" in result.output
