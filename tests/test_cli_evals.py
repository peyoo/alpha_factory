"""tests/test_cli_evals.py — `quant evals` 命令单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import polars as pl
from typer.testing import CliRunner

from alpha_factory.cli.main import app

runner = CliRunner()

# ── 辅助工具 ─────────────────────────────────────────────────────────────────

MOCK_RESULT = pl.DataFrame(
    {
        "factor": ["factor1", "factor2"],
        "ic_mean": [0.045, 0.031],
        "ic_mean_abs": [0.045, 0.031],
        "ic_ir": [0.72, 0.50],
        "ic_ir_abs": [0.72, 0.50],
        "ann_ret": [0.18, 0.12],
        "sharpe": [1.5, 1.1],
        "turnover_est": [0.30, 0.25],
        "direction": [1, 1],
    }
)

MOCK_LAZY = pl.DataFrame({"date": [], "asset": []}).lazy()


def _patch_dp_and_metrics(mock_result: pl.DataFrame = MOCK_RESULT):
    """返回同时 patch DataProvider.load_pool_data 和 batch_full_metrics 的上下文管理器组合。"""
    return (
        patch(
            "alpha_factory.cli.evals.DataProvider",
            return_value=MagicMock(load_pool_data=MagicMock(return_value=MOCK_LAZY)),
        ),
        patch(
            "alpha_factory.cli.evals.batch_full_metrics",
            return_value=mock_result,
        ),
    )


# ── 测试：--expr 输入 ─────────────────────────────────────────────────────────


def test_evals_single_expr():
    dp_ctx, bm_ctx = _patch_dp_and_metrics()
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            ["evals", "-s", "20220101", "--expr", "factor1=ts_mean(AMOUNT,40)"],
        )
    assert result.exit_code == 0, result.output
    assert "factor1" in result.output
    assert "批量评估完成" in result.output


def test_evals_multiple_exprs():
    dp_ctx, bm_ctx = _patch_dp_and_metrics()
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            [
                "evals",
                "-s",
                "20220101",
                "--expr",
                "factor1=ts_mean(AMOUNT,40)",
                "--expr",
                "factor2=rank(CLOSE)",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "2 个因子" in result.output


def test_evals_expr_without_name():
    """无 '=' 的表达式自动命名为 factor_0。"""
    dp_ctx, bm_ctx = _patch_dp_and_metrics(
        pl.DataFrame(
            {
                "factor": ["factor_0"],
                "ic_mean": [0.04],
                "ic_mean_abs": [0.04],
                "ic_ir": [0.6],
                "ic_ir_abs": [0.6],
                "ann_ret": [0.15],
                "sharpe": [1.2],
                "turnover_est": [0.28],
                "direction": [1],
            }
        )
    )
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            ["evals", "-s", "20220101", "--expr", "ts_mean(AMOUNT,40)"],
        )
    assert result.exit_code == 0, result.output


# ── 测试：--csv-file 输入 ─────────────────────────────────────────────────────


def test_evals_csv_file(tmp_path: Path):
    csv = tmp_path / "factors.csv"
    csv.write_text(
        'name,expression\nfactor1,"ts_mean(AMOUNT,40)"\nfactor2,"rank(CLOSE)"\n'
    )

    dp_ctx, bm_ctx = _patch_dp_and_metrics()
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            ["evals", "-s", "20220101", "--csv-file", str(csv)],
        )
    assert result.exit_code == 0, result.output
    assert "factor1" in result.output


def test_evals_csv_custom_cols(tmp_path: Path):
    csv = tmp_path / "factors.csv"
    csv.write_text('因子名,公式\nf1,"ts_mean(AMOUNT,40)"\n')

    dp_ctx, bm_ctx = _patch_dp_and_metrics(
        pl.DataFrame(
            {
                "factor": ["f1"],
                "ic_mean": [0.04],
                "ic_mean_abs": [0.04],
                "ic_ir": [0.6],
                "ic_ir_abs": [0.6],
                "ann_ret": [0.15],
                "sharpe": [1.2],
                "turnover_est": [0.28],
                "direction": [1],
            }
        )
    )
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            [
                "evals",
                "-s",
                "20220101",
                "--csv-file",
                str(csv),
                "--name-col",
                "因子名",
                "--expr-col",
                "公式",
            ],
        )
    assert result.exit_code == 0, result.output


def test_evals_csv_missing_col(tmp_path: Path):
    csv = tmp_path / "bad.csv"
    csv.write_text('wrong_col,expression\nf1,"ts_mean(AMOUNT,40)"\n')

    with patch("alpha_factory.cli.evals.DataProvider"):
        result = runner.invoke(
            app,
            ["evals", "-s", "20220101", "--csv-file", str(csv)],
        )
    assert result.exit_code != 0
    assert "未找到列" in result.output


# ── 测试：无输入报错 ──────────────────────────────────────────────────────────


def test_evals_no_input_errors():
    result = runner.invoke(app, ["evals", "-s", "20220101"])
    assert result.exit_code != 0
    assert "--expr" in result.output or "至少提供" in result.output


# ── 测试：--output CSV 落盘 ───────────────────────────────────────────────────


def test_evals_output_csv(tmp_path: Path):
    out_file = tmp_path / "result.csv"
    dp_ctx, bm_ctx = _patch_dp_and_metrics()
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            [
                "evals",
                "-s",
                "20220101",
                "--expr",
                "factor1=ts_mean(AMOUNT,40)",
                "--output",
                str(out_file),
            ],
        )
    assert result.exit_code == 0, result.output
    assert out_file.exists()
    saved = pl.read_csv(out_file)
    assert "factor" in saved.columns
    assert len(saved) == 2


# ── 测试：--top-n 参数 ────────────────────────────────────────────────────────


def test_evals_top_n(tmp_path: Path):
    dp_ctx, bm_ctx = _patch_dp_and_metrics()
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            [
                "evals",
                "-s",
                "20220101",
                "--expr",
                "factor1=ts_mean(AMOUNT,40)",
                "--top-n",
                "1",
            ],
        )
    assert result.exit_code == 0, result.output
