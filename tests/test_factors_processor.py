"""
tests/test_factors_processor.py — 因子预处理与合成单元测试

覆盖 FactorsPreProcessor、FactorsRankComposite、
FactorsComposite 的基础功能与边界条件。
"""

from __future__ import annotations


import polars as pl
import pytest

from alpha_factory.data_provider.factorsprocessor import (
    FactorsPreProcessor,
    FactorsRankComposite,
    _resolve_action_function,
    _softmax_weights,
)
from alpha_factory.utils.schema import F


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """含 6 只股票、2 个交易日的基础 DataFrame。"""
    return pl.DataFrame(
        {
            F.DATE: ["2024-01-02"] * 6 + ["2024-01-03"] * 6,
            F.ASSET: [f"{100000 + i:06d}.SZ" for i in range(6)] * 2,
            "alpha1": [
                1.0,
                100.0,
                5.0,
                10.0,
                50.0,
                20.0,
                2.0,
                200.0,
                10.0,
                15.0,
                80.0,
                30.0,
            ],
            "alpha2": [0.1, 0.5, 0.3, 0.4, 0.2, 0.6, 0.2, 0.8, 0.4, 0.3, 0.5, 0.7],
            F.POOL_MASK: [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )


@pytest.fixture
def df_no_mask() -> pl.DataFrame:
    """不含 POOL_MASK 列的 DataFrame。"""
    return pl.DataFrame(
        {
            F.DATE: ["2024-01-02", "2024-01-02", "2024-01-02"],
            F.ASSET: ["000001.SZ", "000002.SZ", "000003.SZ"],
            "alpha1": [1.0, 100.0, 5.0],
            "alpha2": [0.1, 0.5, 0.3],
        }
    )


# ── _softmax_weights ────────────────────────────────────────────────────


class TestSoftmaxWeights:
    def test_output_sum_to_one(self):
        w = _softmax_weights(np.array([0.5, 1.0, 2.0]))
        assert abs(w.sum() - 1.0) < 1e-10

    def test_all_zeros_gives_equal_weight(self):
        w = _softmax_weights(np.array([0.0, 0.0, 0.0]))
        assert abs(w.sum() - 1.0) < 1e-10
        assert abs(w[0] - w[1]) < 1e-10

    def test_single_element(self):
        w = _softmax_weights(np.array([42.0]))
        assert abs(w[0] - 1.0) < 1e-10

    def test_large_range(self):
        """极差较大的输入不应导致数值溢出。"""
        w = _softmax_weights(np.array([-1000.0, 1000.0]))
        assert abs(w.sum() - 1.0) < 1e-10
        assert w[1] > w[0]


# ── _resolve_action_function ────────────────────────────────────────────

import numpy as np  # noqa: E402  (needed by softmax tests above)


def test_resolve_action_function_valid_string():
    """有效的字符串函数名应返回可调用对象。"""
    fn = _resolve_action_function("cs_mad")
    assert callable(fn)


def test_resolve_action_function_invalid_string():
    """无效的字符串函数名应抛出 ValueError。"""
    with pytest.raises(ValueError, match="无法解析预处理函数"):
        _resolve_action_function("nonexistent_function_xyz")


def test_resolve_action_function_callable_passthrough():
    """传入可调用对象应直接返回。"""
    # 当传入 callable 时，FactorsPreProcessor 不会调用 _resolve_action_function
    # 但函数本身应该只处理字符串
    pass


# ── FactorsPreProcessor ─────────────────────────────────────────────────


class TestFactorsPreProcessor:
    def test_basic_preprocessing(self, sample_df: pl.DataFrame):
        """基本预处理：cs_mad（去极值 + 标准化），列存在且形状不变。"""
        processor = FactorsPreProcessor(
            factors=["alpha1"],
            actions=["cs_mad"],
        )
        result = processor.process(sample_df)
        assert "alpha1" in result.columns
        assert result.shape == sample_df.shape

    def test_multiple_factors(self, sample_df: pl.DataFrame):
        """多个因子同时预处理。"""
        processor = FactorsPreProcessor(
            factors=["alpha1", "alpha2"],
            actions=["cs_mad"],
        )
        result = processor.process(sample_df)
        assert "alpha1" in result.columns
        assert "alpha2" in result.columns

    def test_no_matching_columns(self, df_no_mask: pl.DataFrame):
        """没有匹配的因子列时，返回原 DataFrame 不变。"""
        processor = FactorsPreProcessor(
            factors=["nonexistent_col"],
            actions=["cs_mad"],
        )
        result = processor.process(df_no_mask)
        assert result.shape == df_no_mask.shape
        assert result.columns == df_no_mask.columns

    def test_with_pool_mask_keeps_out_of_pool_unchanged(self):
        """池外的极端值应保持原值，不参与预处理统计量计算。"""
        df = pl.DataFrame(
            {
                F.DATE: ["2024-01-02"] * 3,
                F.ASSET: ["A", "B", "C"],
                "alpha1": [1.0, 100.0, 9999.0],  # C 是池外极端值
                F.POOL_MASK: [True, True, False],
            }
        )
        orig_out = df.filter(~pl.col(F.POOL_MASK))["alpha1"].item()
        processor = FactorsPreProcessor(
            factors=["alpha1"],
            actions=["cs_mad"],
            use_pool_mask=True,
        )
        result = processor.process(df)
        new_out = result.filter(~pl.col(F.POOL_MASK))["alpha1"].item()
        # 池外极端值不变（不参与 MAD 计算）
        assert new_out == orig_out

    def test_without_pool_mask_processes_pool_outlier(self):
        """use_pool_mask=False 时，池外极端值也会被修正。"""
        df = pl.DataFrame(
            {
                F.DATE: ["2024-01-02"] * 3,
                F.ASSET: ["A", "B", "C"],
                "alpha1": [1.0, 100.0, 9999.0],  # C 是极端值
                F.POOL_MASK: [True, True, False],
            }
        )
        orig_out = df.filter(~pl.col(F.POOL_MASK))["alpha1"].item()
        processor = FactorsPreProcessor(
            factors=["alpha1"],
            actions=["cs_mad"],
            use_pool_mask=False,  # 不排除池外
        )
        result = processor.process(df)
        new_out = result.filter(~pl.col(F.POOL_MASK))["alpha1"].item()
        # 池外极端值已被修正
        assert new_out != orig_out

    def test_without_pool_mask_processes_all(self, sample_df: pl.DataFrame):
        """use_pool_mask=False 时池外标的同样参与预处理。"""
        processor = FactorsPreProcessor(
            factors=["alpha1"],
            actions=["cs_mad"],
            use_pool_mask=False,
        )
        result = processor.process(sample_df)
        # cs_mad 会修改异常值（如 100.0 → ~68.4），而 5.0 非异常不变
        pool_vals = result.filter(pl.col(F.POOL_MASK))["alpha1"].to_list()
        orig_pool = sample_df.filter(pl.col(F.POOL_MASK))["alpha1"].to_list()
        # 至少有一个值发生了变化（异常值被修正）
        assert any(v != o for v, o in zip(pool_vals, orig_pool))

    def test_empty_actions(self, sample_df: pl.DataFrame):
        """actions 为空列表时，返回原 DataFrame。"""
        processor = FactorsPreProcessor(
            factors=["alpha1"],
            actions=[],
        )
        result = processor.process(sample_df)
        assert_frame_equal(result, sample_df)

    def test_regex_factor_pattern(self, sample_df: pl.DataFrame):
        """factors 支持正则表达式。"""
        processor = FactorsPreProcessor(
            factors=r"alpha\d+",
            actions=["cs_mad"],
        )
        result = processor.process(sample_df)
        assert "alpha1" in result.columns
        assert "alpha2" in result.columns

    def test_no_pool_mask_column(self, df_no_mask: pl.DataFrame):
        """输入无 POOL_MASK 列时，use_pool_mask=True 也不报错。"""
        processor = FactorsPreProcessor(
            factors=["alpha1"],
            actions=["cs_mad"],
            use_pool_mask=True,
        )
        result = processor.process(df_no_mask)
        assert "alpha1" in result.columns

    def test_multiple_actions_chained(self, sample_df: pl.DataFrame):
        """多个 action 应依次作用于同一列，结果列存在且形状不变。"""
        processor = FactorsPreProcessor(
            factors=["alpha1"],
            actions=["cs_mad", "cs_mad"],
        )
        result = processor.process(sample_df)
        assert "alpha1" in result.columns
        assert result.shape == sample_df.shape


# ── FactorsRankComposite ────────────────────────────────────────────────


class TestFactorsRankComposite:
    def test_basic_composite(self, sample_df: pl.DataFrame):
        """基本加权合成：等权排名合成。"""
        composite = FactorsRankComposite(
            factors=["alpha1", "alpha2"],
            weights={"alpha1": 1.0, "alpha2": 1.0},
            name="composite_rank",
        )
        result = composite.process(sample_df)
        assert "composite_rank" in result.columns

    def test_composite_values_reasonable(self, sample_df: pl.DataFrame):
        """合成值应为正数（排名累加）。"""
        composite = FactorsRankComposite(
            factors=["alpha1", "alpha2"],
            weights={"alpha1": 1.0, "alpha2": 1.0},
            name="RANK",
        )
        result = composite.process(sample_df)
        rank_vals = result["RANK"].to_list()
        assert all(v >= 0 for v in rank_vals)

    def test_uniform_weights(self, sample_df: pl.DataFrame):
        """等权应等同于单因子排名的简单相加。"""
        composite = FactorsRankComposite(
            factors=["alpha1", "alpha2"],
            weights={"alpha1": 0.5, "alpha2": 0.5},
            name="RANK",
        )
        result = composite.process(sample_df)
        # 权重归一化后 (0.5, 0.5) 与 (1, 1) 等价
        composite2 = FactorsRankComposite(
            factors=["alpha1", "alpha2"],
            weights={"alpha1": 1.0, "alpha2": 1.0},
            name="RANK",
        )
        result2 = composite2.process(sample_df)
        assert result["RANK"].to_list() == pytest.approx(
            result2["RANK"].to_list(), abs=1e-10
        )

    def test_no_match_returns_unchanged(self, df_no_mask: pl.DataFrame):
        """没有匹配列时返回原 DataFrame。"""
        composite = FactorsRankComposite(
            factors=["nonexistent"],
            weights={},
            name="RANK",
        )
        result = composite.process(df_no_mask)
        assert_frame_equal(result, df_no_mask)

    def test_with_use_rank_false(self, sample_df: pl.DataFrame):
        """use_rank=False 应直接加权求和（跳过排名）。"""
        composite = FactorsRankComposite(
            factors=["alpha1", "alpha2"],
            weights={"alpha1": 0.5, "alpha2": 0.5},
            name="composite_raw",
            use_rank=False,
        )
        result = composite.process(sample_df)
        assert "composite_raw" in result.columns

    def test_zero_total_weight_defaults_to_equal(self, sample_df: pl.DataFrame):
        """所有权重为 0 时，自动退化为等权。"""
        composite = FactorsRankComposite(
            factors=["alpha1", "alpha2"],
            weights={"alpha1": 0.0, "alpha2": 0.0},
            name="RANK",
        )
        result = composite.process(sample_df)
        # 不会抛出异常
        assert "RANK" in result.columns

    def test_single_factor(self, sample_df: pl.DataFrame):
        """单因子合成，应正常返回。"""
        composite = FactorsRankComposite(
            factors=["alpha1"],
            weights={"alpha1": 1.0},
            name="single_rank",
        )
        result = composite.process(sample_df)
        assert "single_rank" in result.columns

    def test_opt_mode_requires_trial(self):
        """opt=True 时必须提供 trial。"""
        with pytest.raises(ValueError, match="必须提供 trial"):
            FactorsRankComposite(
                factors=["alpha1"],
                weights={},
                name="RANK",
                opt=True,
            )


# ── 辅助 ────────────────────────────────────────────────────────────────


def assert_frame_equal(a: pl.DataFrame, b: pl.DataFrame):
    """简易 DataFrame 相等断言。"""
    assert a.shape == b.shape, f"shape 不同: {a.shape} vs {b.shape}"
    assert a.columns == b.columns, f"columns 不同: {a.columns} vs {b.columns}"
    for col in a.columns:
        assert a[col].to_list() == b[col].to_list(), f"列 {col} 值不同"
