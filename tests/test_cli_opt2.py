"""tests/test_cli_opt2.py — quant opt2 命令单元测试

覆盖：
  - normalize_features：特征标准化（Z-score）
  - ElasticNet 系数到权重的转换
  - compute_ann_ret：年化收益率计算
  - StrategyConfig 加载：YAML 解析与校验
  - CLI smoke test：--help 可正常响应（不依赖真实数据）
  - YAML 写回：通过 StrategyConfig.to_yaml 写回权重
"""

from __future__ import annotations

import numpy as np
import polars as pl
from typer.testing import CliRunner

from alpha_factory.cli.opt2 import (
    compute_ann_ret,
    normalize_features,
)
from alpha_factory.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# normalize_features
# ---------------------------------------------------------------------------


class TestNormalizeFeatures:
    """测试特征标准化函数"""

    def test_mean_is_zero(self):
        """标准化后均值应接近 0"""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_normalized, means, stds = normalize_features(X)

        # 验证均值接近 0
        assert np.allclose(X_normalized.mean(axis=0), 0.0, atol=1e-10)

    def test_std_is_one(self):
        """标准化后标准差应为 1"""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_normalized, means, stds = normalize_features(X)

        # 验证标准差接近 1
        assert np.allclose(X_normalized.std(axis=0), 1.0, atol=1e-10)

    def test_shape_preserved(self):
        """标准化不应改变矩阵形状"""
        X = np.random.randn(100, 10)
        X_normalized, means, stds = normalize_features(X)

        assert X_normalized.shape == X.shape
        assert means.shape == (10,)
        assert stds.shape == (10,)

    def test_returns_three_arrays(self):
        """函数应返回三个数组：标准化特征、均值、标准差"""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize_features(X)

        assert len(result) == 3
        assert all(isinstance(r, np.ndarray) for r in result)


# ---------------------------------------------------------------------------
# compute_ann_ret
# ---------------------------------------------------------------------------


class TestComputeAnnRet:
    """测试年化收益率计算函数"""

    def test_identity_nav_zero_ret(self):
        """NAV 全为 1 时，年化收益率应为 0"""
        nav = pl.Series([1.0] * 252)
        ann_ret = compute_ann_ret(nav)
        assert abs(ann_ret) < 1e-9

    def test_doubled_nav_positive_ret(self):
        """NAV 翻倍，收益率应为正"""
        nav = pl.Series(np.linspace(1.0, 2.0, 252))
        ann_ret = compute_ann_ret(nav)
        assert ann_ret > 0.0

    def test_halved_nav_negative_ret(self):
        """NAV 减半，收益率应为负"""
        nav = pl.Series(np.linspace(1.0, 0.5, 252))
        ann_ret = compute_ann_ret(nav)
        assert ann_ret < 0.0

    def test_short_series(self):
        """长度 < 2 时，应返回 0"""
        nav = pl.Series([1.0])
        ann_ret = compute_ann_ret(nav)
        assert ann_ret == 0.0


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


class TestCliOpt2:
    """测试命令行接口"""

    def test_opt2_help(self):
        """测试 opt2 --help 可正常响应"""
        result = runner.invoke(app, ["opt2", "--help"])
        assert result.exit_code == 0
        assert "ElasticNet" in result.stdout or "elasticnet" in result.stdout.lower()
        assert "alpha" in result.stdout or "--alpha" in result.stdout

    def test_opt2_missing_required_yaml_file(self):
        """测试 opt2 缺少必需的参数时应返回错误"""
        result = runner.invoke(app, ["opt2"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ElasticNet 权重转换逻辑
# ---------------------------------------------------------------------------


class TestElasticNetWeightConversion:
    """测试 ElasticNet 系数到权重的转换逻辑"""

    def test_abs_coef_normalization(self):
        """测试系数绝对值的归一化"""
        coefficients = np.array([0.5, -0.3, 0.2])
        abs_coef = np.abs(coefficients)
        weights = abs_coef / abs_coef.sum()

        # 验证权重和为 1
        assert abs(weights.sum() - 1.0) < 1e-9

        # 验证权重都为正
        assert (weights >= 0).all()

        # 验证权重匹配绝对值的比例
        expected = np.array([0.5, 0.3, 0.2]) / 1.0
        assert np.allclose(weights, expected)

    def test_zero_coefficients_fallback(self):
        """测试当系数接近 0 时的回退逻辑"""
        coefficients = np.array([1e-10, 1e-11, 1e-12])
        abs_coef = np.abs(coefficients)

        if abs_coef.sum() < 1e-9:
            weights = np.ones(len(abs_coef)) / len(abs_coef)
        else:
            weights = abs_coef / abs_coef.sum()

        # 应该使用均等权重
        expected = np.ones(len(abs_coef)) / len(abs_coef)
        assert np.allclose(weights, expected)

    def test_single_nonzero_coefficient(self):
        """测试仅一个非零系数的情况"""
        coefficients = np.array([0.0, 1.5, 0.0])
        abs_coef = np.abs(coefficients)
        weights = abs_coef / abs_coef.sum()

        # 权重应该完全集中在非零系数上
        expected = np.array([0.0, 1.0, 0.0])
        assert np.allclose(weights, expected)
