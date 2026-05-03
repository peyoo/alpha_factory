"""tests/test_cli_opt.py — quant opt 命令单元测试

覆盖：
  - softmax_weights：归一化约束（sum=1, w≥0）
  - FactorsRankComposite.process：合成与加权逻辑（替代原 make_composite 测试）
  - compute_ann_ret：年化收益率计算
  - StrategyConfig 加载：YAML 解析与校验（替代原 _extract_ranks 测试）
  - CLI smoke test：--help 可正常响应（不依赖真实数据）
  - YAML 写回：通过 StrategyConfig.to_yaml 写回权重
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from alpha_factory.cli.opt import (
    compute_ann_ret,
    softmax_weights,
)
from alpha_factory.data_provider.factorsprocessor import FactorsRankComposite
from alpha_factory.cli.main import app
from alpha_factory.config.strategy import StrategyConfig

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """去除 rich/click 输出的 ANSI 转义码以便做纯文本匹配。"""
    import re

    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


# ---------------------------------------------------------------------------
# softmax_weights
# ---------------------------------------------------------------------------


class TestSoftmaxWeights:
    def test_sum_is_one(self):
        x = np.array([0.5, -1.0, 2.0, 0.0])
        w = softmax_weights(x)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_all_positive(self):
        x = np.array([-3.0, -3.0, -3.0])
        w = softmax_weights(x)
        assert (w > 0).all()

    def test_uniform_input_uniform_output(self):
        x = np.zeros(4)
        w = softmax_weights(x)
        assert np.allclose(w, 0.25)

    def test_numerical_stability_large_values(self):
        x = np.array([1000.0, 999.9, 0.0])
        w = softmax_weights(x)
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= 0).all()


# ---------------------------------------------------------------------------
# make_composite
# ---------------------------------------------------------------------------


class TestMakeComposite:
    """TestMakeComposite - 验证因子列的加权合成逻辑（使用 FactorsRankComposite.process）。"""

    def _make_base_df(self, n_days: int = 4, n_stocks: int = 3):
        """Minimal DataFrame with factor columns (w/o _RANK_ prefix)."""
        import itertools
        from alpha_factory.utils.schema import F

        dates = [f"2023-01-0{i + 1}" for i in range(n_days)]
        stocks = [f"00000{i}.SZ" for i in range(n_stocks)]
        rows = list(itertools.product(dates, stocks))
        n = len(rows)
        return pl.DataFrame(
            {
                F.DATE: [r[0] for r in rows],
                F.ASSET: [r[1] for r in rows],
                "f1": np.arange(1, n + 1, dtype=float),
                "f2": np.arange(n, 0, -1, dtype=float),
            }
        )

    def test_composite_column_created(self):
        df = self._make_base_df()
        result = FactorsRankComposite(
            factors=["f1", "f2"], weights={"f1": 0.5, "f2": 0.5}, name="COMPOSITE_OPT"
        ).process(df)
        assert "COMPOSITE_OPT" in result.columns

    def test_weighted_sum_correctness(self):
        """composite = w1*rank(f1) + w2*rank(f2)，descending rank 使大值获得 rank 1。"""
        df = self._make_base_df(n_days=1, n_stocks=2)
        # ascending=False（默认）→ descending rank:
        # f1=[1,2] → rank=[2,1], f2=[2,1] → rank=[1,2], w={f1:0.8, f2:0.2}
        result = FactorsRankComposite(
            factors=["f1", "f2"], weights={"f1": 0.8, "f2": 0.2}, name="COMPOSITE_OPT"
        ).process(df.head(2))
        expected = [
            0.8 * 2 + 0.2 * 1,  # 1.8
            0.8 * 1 + 0.2 * 2,  # 1.2
        ]
        assert result["COMPOSITE_OPT"].to_list() == pytest.approx(expected, rel=1e-6)

    def test_pure_first_factor(self):
        """weights={f1:1, f2:0} 时 composite == descending rank of f1。"""
        df = self._make_base_df(n_days=2, n_stocks=2)
        result = FactorsRankComposite(
            factors=["f1", "f2"], weights={"f1": 1.0, "f2": 0.0}, name="COMPOSITE_OPT"
        ).process(df)
        # ascending=False（默认）→ descending rank: 大值获 rank 1
        # day1: f1=[1,2] → rank=[2,1]; day2: f1=[3,4] → rank=[2,1]
        expected_ranks = [2.0, 1.0, 2.0, 1.0]
        assert result["COMPOSITE_OPT"].to_list() == pytest.approx(
            expected_ranks, rel=1e-6
        )

    def test_original_df_not_mutated(self):
        df = self._make_base_df()
        _ = FactorsRankComposite(
            factors=["f1", "f2"], weights={"f1": 0.5, "f2": 0.5}, name="COMPOSITE_OPT"
        ).process(df)
        assert "COMPOSITE_OPT" not in df.columns


# ---------------------------------------------------------------------------
# compute_ann_ret
# ---------------------------------------------------------------------------


class TestComputeAnnRet:
    def test_flat_nav_returns_zero(self):
        nav = pl.Series([1.0] * 252)
        assert compute_ann_ret(nav) == pytest.approx(0.0, abs=1e-9)

    def test_positive_return(self):
        # NAV 从 1 涨到 2，共 252 天 → 年化约 100%
        nav = pl.Series([1.0 + i / 252.0 for i in range(253)])
        ann_ret = compute_ann_ret(nav)
        assert ann_ret > 0

    def test_too_short_nav(self):
        nav = pl.Series([1.0])
        assert compute_ann_ret(nav) == 0.0

    def test_empty_nav(self):
        nav = pl.Series([], dtype=pl.Float64)
        assert compute_ann_ret(nav) == 0.0


# ---------------------------------------------------------------------------
# StrategyConfig 加载（替代原 _extract_ranks 测试）
# ---------------------------------------------------------------------------


class TestStrategyConfigLoad:
    """通过 StrategyConfig.from_yaml 验证 YAML 解析与校验。"""

    def _make_yaml_content(self, n_ranks: int = 2) -> str:
        ranks_block = "\n".join(
            f"  - name: f{i}\n"
            f"    expression: CLOSE.shift({i})\n"
            f"    direction: 1\n"
            f"    weight: 1.0"
            for i in range(1, n_ranks + 1)
        )
        return f"name: s1\npool: main_small_pool\nranks:\n{ranks_block}\n"

    def test_load_name_and_ranks(self, tmp_path):
        yaml_file = tmp_path / "s1.yaml"
        yaml_file.write_text(self._make_yaml_content(2), encoding="utf-8")
        cfg = StrategyConfig.from_yaml(yaml_file)
        assert cfg.name == "s1"
        assert len(cfg.ranks) == 2

    def test_load_three_ranks(self, tmp_path):
        yaml_file = tmp_path / "s1.yaml"
        yaml_file.write_text(self._make_yaml_content(3), encoding="utf-8")
        cfg = StrategyConfig.from_yaml(yaml_file)
        assert len(cfg.ranks) == 3

    def test_defaults_applied(self, tmp_path):
        """未指定字段应取 StrategyConfig 默认值。"""
        yaml_file = tmp_path / "s1.yaml"
        yaml_file.write_text(self._make_yaml_content(2), encoding="utf-8")
        cfg = StrategyConfig.from_yaml(yaml_file)
        assert cfg.pool == "main_small_pool"
        assert cfg.cost == pytest.approx(0.003)
        assert cfg.hold_num == 10

    def test_factor_names_property(self, tmp_path):
        yaml_file = tmp_path / "s1.yaml"
        yaml_file.write_text(self._make_yaml_content(2), encoding="utf-8")
        cfg = StrategyConfig.from_yaml(yaml_file)
        # 所有因子名字在 from_yaml 时统一生成为 rank_f1、rank_f2...
        assert cfg.factor_names == ["rank_f1", "rank_f2"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            StrategyConfig.from_yaml(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCliSmoke:
    def test_opt_help(self):
        result = runner.invoke(app, ["opt", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output).lower()
        assert "yaml" in output
        assert "n-trials" in output

    def test_opt_missing_yaml_fails(self, tmp_path):
        nonexistent = str(tmp_path / "no_such_file.yaml")
        result = runner.invoke(app, ["opt", "--yaml", nonexistent])
        # 应以非零退出码结束（文件不存在）
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# YAML 写回测试
# ---------------------------------------------------------------------------


class TestYamlWriteback:
    """YAML 写回测试（通过 StrategyConfig.to_yaml）。"""

    def _make_yaml_content(self) -> str:
        return (
            "name: test_strat\n"
            "pool: main_small_pool\n"
            "ranks:\n"
            "  - name: f1\n"
            "    weight: 1.0\n"
            "    expression: CLOSE.shift(1)\n"
            "    direction: 1\n"
            "  - name: f2\n"
            "    weight: 1.0\n"
            "    expression: ts_mean(AMOUNT, 5)\n"
            "    direction: -1\n"
        )

    def test_weight_updated_in_yaml(self, tmp_path):
        yaml_file = tmp_path / "test_strat.yaml"
        yaml_file.write_text(self._make_yaml_content(), encoding="utf-8")

        # 模拟：通过 StrategyConfig 更新权重并写回
        new_weights = np.array([0.3, 0.7])
        cfg = StrategyConfig.from_yaml(yaml_file)
        for rank_item, w in zip(cfg.ranks, new_weights):
            rank_item.weight = round(float(w), 6)
        cfg.to_yaml(yaml_file)

        # 读回验证
        cfg2 = StrategyConfig.from_yaml(yaml_file)
        assert abs(cfg2.ranks[0].weight - 0.3) < 1e-6
        assert abs(cfg2.ranks[1].weight - 0.7) < 1e-6

    def test_structure_preserved_after_writeback(self, tmp_path):
        yaml_file = tmp_path / "test_strat2.yaml"
        yaml_file.write_text(self._make_yaml_content(), encoding="utf-8")

        # 更新 weight，然后写回
        cfg = StrategyConfig.from_yaml(yaml_file)
        for rank_item in cfg.ranks:
            rank_item.weight = 0.5
        cfg.to_yaml(yaml_file)

        cfg2 = StrategyConfig.from_yaml(yaml_file)
        # expression 和 direction 保持不变
        assert cfg2.ranks[0].expression == "CLOSE.shift(1)"
        assert cfg2.ranks[1].direction == -1
        # 名字在 from_yaml 时统一生成为 rank_f1、rank_f2...
        assert cfg2.ranks[0].name == "rank_f1"
