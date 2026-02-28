"""
tests/test_cli_ml.py
====================
单元测试：quant ml 因子筛选模块

核心逻辑：run_regularized_selection 基于面板数据回归。
不依赖真实数据，通过 mock 面板 DataFrame 覆盖核心逻辑。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from alpha_factory.ml.dim_reduction import (
    FactorSelectionResult,
    build_metrics_matrix,
    detect_metric_cols,
    load_factor_csv,
    run_regularized_selection,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


def _make_eval_df(n: int = 20) -> pl.DataFrame:
    """构造一个模拟 evals 结果 DataFrame（n 个因子）。"""
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "factor": [f"f{i}" for i in range(n)],
            "expression": [f"CLOSE.rolling_mean({i + 1})" for i in range(n)],
            "ic_mean": rng.normal(0, 0.01, n).tolist(),
            "ic_mean_abs": rng.uniform(0.005, 0.02, n).tolist(),
            "ic_ir": rng.normal(0, 0.3, n).tolist(),
            "ic_ir_abs": rng.uniform(0.1, 0.5, n).tolist(),
            "ann_ret": rng.uniform(0.1, 0.8, n).tolist(),
            "sharpe": rng.uniform(0.5, 2.5, n).tolist(),
            "turnover_est": rng.uniform(0.01, 0.05, n).tolist(),
            "direction": rng.choice([-1, 1], size=n).tolist(),
        }
    )


def _make_panel_df(
    n_dates: int = 50,
    n_assets: int = 100,
    n_factors: int = 10,
) -> pl.DataFrame:
    """构造模拟面板数据（DATE x ASSET x factors + LABEL_FOR_IC + POOL_MASK）。

    目标 = f0*0.5 + f1*(-0.3) + f2*0.2 + noise，确保稀疏信号可被正则化检测。
    """
    rng = np.random.default_rng(42)
    n_rows = n_dates * n_assets

    dates = []
    assets = []
    for d in range(n_dates):
        date_str = f"2024{(d // 28 + 1):02d}{(d % 28 + 1):02d}"
        for a in range(n_assets):
            dates.append(date_str)
            assets.append(f"asset_{a:04d}")

    data: dict[str, list] = {
        "DATE": dates,
        "ASSET": assets,
        "POOL_MASK": [True] * n_rows,
    }

    # 生成因子值
    factor_cols = [f"f{i}" for i in range(n_factors)]
    factor_matrix = rng.normal(0, 1, (n_rows, n_factors))
    for i, col in enumerate(factor_cols):
        data[col] = factor_matrix[:, i].tolist()

    # 目标 = 部分因子的线性组合 + 噪声（前 3 个因子有真实信号）
    true_coefs = np.zeros(n_factors)
    true_coefs[0] = 0.5
    true_coefs[1] = -0.3
    true_coefs[2] = 0.2
    label = factor_matrix @ true_coefs + rng.normal(0, 0.5, n_rows)
    data["LABEL_FOR_IC"] = label.tolist()

    return pl.DataFrame(data)


# ──────────────────────────────────────────────
# detect_metric_cols
# ──────────────────────────────────────────────


def test_detect_metric_cols_standard():
    """标准 evals CSV：应该返回已知数值指标列，不含元数据列。"""
    df = _make_eval_df()
    cols = detect_metric_cols(df)
    for meta in ("factor", "expression", "direction"):
        assert meta not in cols, f"{meta} 不应出现在指标列中"
    for expected in ("ic_mean", "ic_ir", "ann_ret", "sharpe"):
        assert expected in cols, f"{expected} 应在指标列中"


def test_detect_metric_cols_minimal():
    """只有 2 列的 DataFrame：返回唯一的数值列。"""
    df = pl.DataFrame({"factor": ["f1", "f2"], "ic_mean": [0.01, -0.01]})
    cols = detect_metric_cols(df)
    assert cols == ["ic_mean"]


# ──────────────────────────────────────────────
# build_metrics_matrix
# ──────────────────────────────────────────────


def test_build_metrics_matrix_shape():
    n = 15
    df = _make_eval_df(n)
    metric_cols = detect_metric_cols(df)
    X, factor_names, feat_names = build_metrics_matrix(df, metric_cols)
    assert X.shape == (n, len(metric_cols)), "矩阵行数应等于因子数"
    assert len(factor_names) == n
    assert feat_names == metric_cols


def test_build_metrics_matrix_drops_nan():
    """含 NaN 行应被自动丢弃。"""
    df = pl.DataFrame(
        {
            "factor": ["f1", "f2", "f3"],
            "ic_mean": [0.01, None, 0.02],
            "ann_ret": [0.5, 0.3, 0.4],
        }
    )
    X, factor_names, _ = build_metrics_matrix(df, ["ic_mean", "ann_ret"])
    assert X.shape[0] == 2, "含 NaN 的行应被移除"
    assert "f2" not in factor_names


# ──────────────────────────────────────────────
# load_factor_csv
# ──────────────────────────────────────────────


def test_load_factor_csv_roundtrip(tmp_path: Path):
    """将 DataFrame 写到临时 CSV，再读回来应一致。"""
    df = _make_eval_df(10)
    csv_path = tmp_path / "test_factors.csv"
    df.write_csv(csv_path)

    loaded = load_factor_csv(csv_path)
    assert loaded.shape == df.shape
    assert loaded.columns == df.columns


# ──────────────────────────────────────────────
# run_regularized_selection - Lasso
# ──────────────────────────────────────────────


def test_lasso_basic():
    """基本 Lasso 运行：返回 FactorSelectionResult，selected_factors 非空。"""
    panel = _make_panel_df()
    factor_cols = [f"f{i}" for i in range(10)]
    result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        method="lasso",
    )
    assert isinstance(result, FactorSelectionResult)
    assert result.method == "lasso"
    assert result.l1_ratio == 1.0
    assert len(result.selected_factors) >= 1
    assert len(result.factor_names) == 10


def test_lasso_sparsity():
    """Lasso 应将部分无关因子系数压为 0。"""
    panel = _make_panel_df(n_factors=10)
    factor_cols = [f"f{i}" for i in range(10)]
    result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        method="lasso",
    )
    assert len(result.eliminated_factors) >= 1, "Lasso 应剔除至少一个无效因子"


def test_lasso_coefs_count():
    """所有因子都应有系数。"""
    panel = _make_panel_df(n_factors=8)
    factor_cols = [f"f{i}" for i in range(8)]
    result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        method="lasso",
    )
    assert len(result.factor_coefs) == 8
    assert set(result.factor_coefs.keys()) == set(factor_cols)


def test_lasso_selected_eliminated_complement():
    """selected + eliminated 应覆盖所有因子。"""
    panel = _make_panel_df(n_factors=10)
    factor_cols = [f"f{i}" for i in range(10)]
    result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        method="lasso",
    )
    all_factors = set(result.selected_factors) | set(result.eliminated_factors)
    assert all_factors == set(factor_cols)


def test_lasso_r2_range():
    """R2 应在合理范围内。"""
    panel = _make_panel_df(n_factors=10)
    factor_cols = [f"f{i}" for i in range(10)]
    result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        method="lasso",
    )
    assert result.r2 <= 1.0 + 1e-6


def test_lasso_invalid_target():
    """不存在的 target_col 应抛出 ValueError。"""
    panel = _make_panel_df(n_factors=5)
    factor_cols = [f"f{i}" for i in range(5)]
    with pytest.raises(ValueError, match="target_col"):
        run_regularized_selection(
            panel,
            factor_cols,
            target_col="nonexistent_col",
            method="lasso",
        )


# ──────────────────────────────────────────────
# run_regularized_selection - Elastic Net
# ──────────────────────────────────────────────


def test_elastic_net_basic():
    """Elastic Net 基本运行。"""
    panel = _make_panel_df(n_factors=10)
    factor_cols = [f"f{i}" for i in range(10)]
    result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        method="elastic-net",
        l1_ratio=0.5,
    )
    assert isinstance(result, FactorSelectionResult)
    assert result.method == "elastic-net"
    assert len(result.selected_factors) >= 1


def test_elastic_net_less_sparse_than_lasso():
    """Elastic Net（l1_ratio<1）通常保留不少于 Lasso 的因子数。"""
    panel = _make_panel_df(n_factors=10, n_dates=80, n_assets=100)
    factor_cols = [f"f{i}" for i in range(10)]
    lasso_result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        method="lasso",
    )
    enet_result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        method="elastic-net",
        l1_ratio=0.5,
    )
    assert len(enet_result.selected_factors) >= len(lasso_result.selected_factors) - 2


def test_invalid_method():
    """不支持的 method 应抛异常。"""
    panel = _make_panel_df(n_factors=3)
    factor_cols = [f"f{i}" for i in range(3)]
    with pytest.raises(ValueError, match="method"):
        run_regularized_selection(
            panel,
            factor_cols,
            method="ridge",  # type: ignore[arg-type]
        )


def test_pool_mask_filtering():
    """设置 pool_mask=False 的行应被过滤。"""
    panel = _make_panel_df(n_dates=20, n_assets=50, n_factors=5)
    n = len(panel)
    masks = [i % 2 == 0 for i in range(n)]
    panel = panel.with_columns(pl.Series("POOL_MASK", masks))

    factor_cols = [f"f{i}" for i in range(5)]
    result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
    )
    assert result.n_samples < n
    assert result.n_samples > 0


def test_no_pool_mask():
    """pool_mask_col=None 不应过滤任何行。"""
    panel = _make_panel_df(n_dates=20, n_assets=50, n_factors=5)
    factor_cols = [f"f{i}" for i in range(5)]
    result = run_regularized_selection(
        panel,
        factor_cols,
        target_col="LABEL_FOR_IC",
        pool_mask_col=None,
    )
    assert result.n_samples == len(panel)


# ──────────────────────────────────────────────
# CLI 集成（typer CliRunner + mock DataProvider）
# ──────────────────────────────────────────────


def test_cli_ml_missing_csv(tmp_path: Path):
    """--csv 指向不存在的文件应退出码非 0。"""
    from typer.testing import CliRunner

    from alpha_factory.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ml",
            "--csv",
            str(tmp_path / "nonexistent.csv"),
            "--start-date",
            "20240101",
            "--no-report",
            "--no-save",
        ],
    )
    assert result.exit_code != 0 or "不存在" in (result.output or "")


def test_cli_ml_lasso_runs(tmp_path: Path):
    """CLI Lasso 完整运行（Mock DataProvider）。"""
    from typer.testing import CliRunner

    from alpha_factory.cli.main import app

    n_factors = 5
    csv_path = tmp_path / "factors.csv"
    evals_df = pl.DataFrame(
        {
            "factor": [f"f{i}" for i in range(n_factors)],
            "expression": [f"CLOSE + {i}" for i in range(n_factors)],
            "sharpe": [0.5 + i * 0.1 for i in range(n_factors)],
        }
    )
    evals_df.write_csv(csv_path)

    panel = _make_panel_df(n_dates=30, n_assets=50, n_factors=n_factors)
    mock_lf = MagicMock()
    mock_lf.collect.return_value = panel

    with patch("alpha_factory.cli.ml.DataProvider") as mock_dp_cls:
        mock_dp_cls.return_value.load_pool_data.return_value = mock_lf
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "ml",
                "--csv",
                str(csv_path),
                "--method",
                "lasso",
                "--start-date",
                "20240101",
                "--target",
                "LABEL_FOR_IC",
                "--no-report",
                "--save",
                "--output-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, f"CLI 退出码非 0:\n{result.output}"
    saved_files = list(tmp_path.glob("ml_lasso_*.csv"))
    assert len(saved_files) == 1, "应有 1 个 Lasso 筛选结果 CSV"
    loaded = pl.read_csv(saved_files[0])
    assert "factor" in loaded.columns
    assert "coefficient" in loaded.columns
    assert "selected" in loaded.columns
    assert len(loaded) == n_factors


def test_cli_ml_elastic_net_runs(tmp_path: Path):
    """CLI Elastic Net 完整运行（Mock DataProvider）。"""
    from typer.testing import CliRunner

    from alpha_factory.cli.main import app

    n_factors = 5
    csv_path = tmp_path / "factors.csv"
    evals_df = pl.DataFrame(
        {
            "factor": [f"f{i}" for i in range(n_factors)],
            "expression": [f"CLOSE + {i}" for i in range(n_factors)],
            "sharpe": [0.5 + i * 0.1 for i in range(n_factors)],
        }
    )
    evals_df.write_csv(csv_path)

    panel = _make_panel_df(n_dates=30, n_assets=50, n_factors=n_factors)
    mock_lf = MagicMock()
    mock_lf.collect.return_value = panel

    with patch("alpha_factory.cli.ml.DataProvider") as mock_dp_cls:
        mock_dp_cls.return_value.load_pool_data.return_value = mock_lf
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "ml",
                "--method",
                "elastic-net",
                "--l1-ratio",
                "0.5",
                "--csv",
                str(csv_path),
                "--start-date",
                "20240101",
                "--target",
                "LABEL_FOR_IC",
                "--no-report",
                "--save",
                "--output-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, f"CLI 退出码非 0:\n{result.output}"
    saved_files = list(tmp_path.glob("ml_elastic_net_*.csv"))
    assert len(saved_files) == 1
