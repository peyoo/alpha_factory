"""tests/test_cli_evals.py — `quant evals` 命令单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

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
        "ann_ret": [0.28, 0.22],
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


def test_evals_single_expr_without_start_date():
    dp_ctx, bm_ctx = _patch_dp_and_metrics()
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            ["evals", "--expr", "factor1=ts_mean(AMOUNT,40)"],
        )
    assert result.exit_code == 0, result.output
    assert "factor1" in result.output


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


# ── 测试：无输入且 pool 目录为空时报错 ────────────────────────────────


def test_evals_no_input_errors(tmp_path: Path):
    """pool 目录为空 + 无 --expr / --csv 时应内退并打印提示。"""
    from alpha_factory.data_provider.pool import MainSmallPool

    with patch.object(
        MainSmallPool, "pool_dir", new_callable=PropertyMock, return_value=tmp_path
    ):
        result = runner.invoke(app, ["evals", "-s", "20220101"])
    assert result.exit_code != 0
    assert "--expr" in result.output or "至少提供" in result.output


# ── 测试：自动扫描 pool 目录模式 ────────────────────────────────────


def test_evals_auto_pool_mode(tmp_path: Path):
    """未指定 --csv / --expr 时，自动扫描 pool 目录内所有 CSV，合并去重后评估并落盘。"""
    from alpha_factory.data_provider.pool import MainSmallPool
    import polars as pl

    # 在 tmp_path 下创建两个 CSV
    csv_a = tmp_path / "pool_a.csv"
    csv_b = tmp_path / "pool_b.csv"
    pl.DataFrame(
        {"factor": ["f1", "f2"], "expression": ["ts_mean(AMOUNT,5)", "cs_rank(CLOSE)"]}
    ).write_csv(csv_a)
    pl.DataFrame(
        {
            "factor": ["f2", "f3"],
            "expression": ["cs_rank(CLOSE_other)", "ts_std_dev(CLOSE,10)"],
        }
    ).write_csv(csv_b)
    # f2 在两个文件中都有，加载后应去重为 3 个因子

    dp_ctx, bm_ctx = _patch_dp_and_metrics(
        mock_result=pl.DataFrame(
            {
                "factor": ["f1", "f2", "f3"],
                "ic_mean": [0.05, 0.04, 0.03],
                "ic_mean_abs": [0.05, 0.04, 0.03],
                "ic_ir": [0.8, 0.7, 0.6],
                "ic_ir_abs": [0.8, 0.7, 0.6],
                "ann_ret": [0.30, 0.25, 0.22],
                "sharpe": [1.8, 1.5, 1.2],
                "turnover_est": [0.28, 0.30, 0.25],
                "direction": [1, 1, 1],
            }
        )
    )
    with (
        patch.object(
            MainSmallPool, "pool_dir", new_callable=PropertyMock, return_value=tmp_path
        ),
        dp_ctx,
        bm_ctx,
    ):
        result = runner.invoke(app, ["evals", "-s", "20220101"])

    assert result.exit_code == 0, result.output
    # 应自动扫描提示
    assert "扫描池目录" in result.output or "自动" in result.output
    # 输出文件应为 <pool_dir>/main_small_pool.csv
    expected_out = tmp_path / "main_small_pool.csv"
    assert expected_out.exists(), f"期望输出文件不存在: {expected_out}"
    saved = pl.read_csv(expected_out)
    assert "factor" in saved.columns


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


def test_evals_output_csv_does_not_contain_timing_columns(tmp_path: Path):
    out_file = tmp_path / "result_with_timing.csv"
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
    saved = pl.read_csv(out_file)
    assert "factor_eval_seconds" not in saved.columns
    assert "total_eval_seconds" not in saved.columns
    assert "时间统计" in result.output


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


def test_evals_default_quality_filter(tmp_path: Path):
    low_quality = pl.DataFrame(
        {
            "factor": ["factor1", "factor2"],
            "ic_mean": [0.02, 0.01],
            "ic_mean_abs": [0.02, 0.01],
            "ic_ir": [0.3, 0.2],
            "ic_ir_abs": [0.3, 0.2],
            "ann_ret": [0.19, 0.12],
            "sharpe": [0.9, 0.8],
            "turnover_est": [0.30, 0.25],
            "direction": [1, 1],
        }
    )
    dp_ctx, bm_ctx = _patch_dp_and_metrics(low_quality)
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            [
                "evals",
                "-s",
                "20220101",
                "--expr",
                "factor1=ts_mean(AMOUNT,40)",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "过滤后无可用因子" in result.output


def test_evals_relaxed_quality_filter(tmp_path: Path):
    low_quality = pl.DataFrame(
        {
            "factor": ["factor1"],
            "ic_mean": [0.02],
            "ic_mean_abs": [0.02],
            "ic_ir": [0.3],
            "ic_ir_abs": [0.3],
            "ann_ret": [0.12],
            "sharpe": [0.8],
            "turnover_est": [0.30],
            "direction": [1],
        }
    )
    dp_ctx, bm_ctx = _patch_dp_and_metrics(low_quality)
    with dp_ctx, bm_ctx:
        result = runner.invoke(
            app,
            [
                "evals",
                "-s",
                "20220101",
                "--expr",
                "factor1=ts_mean(AMOUNT,40)",
                "--min-sharpe",
                "0.5",
                "--min-ann-ret",
                "0.1",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "factor1" in result.output


def test_evals_batch_size_chunking():
    dp_ctx, _ = _patch_dp_and_metrics()
    side_effect_results = [
        pl.DataFrame(
            {
                "factor": ["f1", "f2"],
                "ic_mean": [0.05, 0.04],
                "ic_mean_abs": [0.05, 0.04],
                "ic_ir": [0.7, 0.6],
                "ic_ir_abs": [0.7, 0.6],
                "ann_ret": [0.3, 0.25],
                "sharpe": [1.4, 1.2],
                "turnover_est": [0.2, 0.2],
                "direction": [1, 1],
            }
        ),
        pl.DataFrame(
            {
                "factor": ["f3"],
                "ic_mean": [0.03],
                "ic_mean_abs": [0.03],
                "ic_ir": [0.5],
                "ic_ir_abs": [0.5],
                "ann_ret": [0.22],
                "sharpe": [1.1],
                "turnover_est": [0.2],
                "direction": [1],
            }
        ),
    ]

    with (
        dp_ctx,
        patch(
            "alpha_factory.cli.evals.batch_full_metrics",
            side_effect=side_effect_results,
        ) as mocked_metrics,
    ):
        result = runner.invoke(
            app,
            [
                "evals",
                "--expr",
                "f1=ts_mean(AMOUNT,5)",
                "--expr",
                "f2=ts_mean(AMOUNT,10)",
                "--expr",
                "f3=ts_mean(AMOUNT,20)",
                "--batch-size",
                "2",
                "--min-sharpe",
                "0",
                "--min-ann-ret",
                "0",
            ],
        )

    assert result.exit_code == 0, result.output
    assert mocked_metrics.call_count == 2
    assert mocked_metrics.call_args_list[0].kwargs["factors"] == ["f1", "f2"]
    assert mocked_metrics.call_args_list[1].kwargs["factors"] == ["f3"]
