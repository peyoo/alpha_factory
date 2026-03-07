"""
tests/test_cli_opt.py — quant opt 命令单元测试

覆盖：
  - softmax_weights：归一化约束（sum=1, w≥0）
  - build_composite_expr：合成表达式格式正确
  - compute_ann_ret：年化收益率计算
  - _extract_ranks：YAML 解析与错误处理
  - CLI smoke test：--help 可正常响应（不依赖真实数据）
  - YAML 写回：weight 字段正确更新
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from click.exceptions import Exit as ClickExit
from typer.testing import CliRunner

from alpha_factory.cli.opt import (
    _extract_ranks,
    compute_ann_ret,
    make_composite,
    softmax_weights,
)
from alpha_factory.cli.main import app

runner = CliRunner()


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
    """TestMakeComposite - 验证预计算列的加权合成逻辑。"""

    def _make_base_df(self, n_days: int = 4, n_stocks: int = 3):
        """Minimal DataFrame with _RANK_ columns mimicking precomputed data."""
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
                "_RANK_f1": np.arange(1, n + 1, dtype=float),
                "_RANK_f2": np.arange(n, 0, -1, dtype=float),
            }
        )

    def test_composite_column_created(self):
        df = self._make_base_df()
        result = make_composite(df, ["f1", "f2"], np.array([0.5, 0.5]))
        assert "COMPOSITE_OPT" in result.columns

    def test_weighted_sum_correctness(self):
        """composite = w1*_RANK_f1 + w2*_RANK_f2，方向已在预计算阶段吸收到 rank 排序方向中。"""
        df = self._make_base_df(n_days=1, n_stocks=2)
        # _RANK_f1=[1,2], _RANK_f2=[2,1], w=[0.8, 0.2]
        result = make_composite(df.head(2), ["f1", "f2"], np.array([0.8, 0.2]))
        expected = [
            0.8 * 1 + 0.2 * 2,  # 1.2
            0.8 * 2 + 0.2 * 1,  # 1.8
        ]
        assert result["COMPOSITE_OPT"].to_list() == pytest.approx(expected, rel=1e-6)

    def test_pure_first_factor(self):
        """When w=[1,0], composite == _RANK_f1（direction 已内化于 rank 列）."""
        df = self._make_base_df(n_days=2, n_stocks=2)
        result = make_composite(df, ["f1", "f2"], np.array([1.0, 0.0]))
        assert result["COMPOSITE_OPT"].to_list() == pytest.approx(
            result["_RANK_f1"].to_list(), rel=1e-6
        )

    def test_original_df_not_mutated(self):
        df = self._make_base_df()
        _ = make_composite(df, ["f1", "f2"], np.array([0.5, 0.5]))
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
# _extract_ranks
# ---------------------------------------------------------------------------


class TestExtractRanks:
    """TestExtractRanks - 验证扁平 YAML 解析。"""

    def _make_data(self, n_ranks: int = 2) -> dict:
        """Generate flat YAML dict with n_ranks ranks."""
        return {
            "name": "s1",
            "type": "single",
            "ranks": [
                {
                    "name": f"f{i}",
                    "expression": f"CLOSE.shift({i})",
                    "direction": 1,
                    "weight": 1.0,
                }
                for i in range(1, n_ranks + 1)
            ],
        }

    def test_extract_name_and_ranks(self):
        data = self._make_data(2)
        name, ranks = _extract_ranks(data)
        assert name == "s1"
        assert len(ranks) == 2

    def test_extract_three_ranks(self):
        data = self._make_data(3)
        _, ranks = _extract_ranks(data)
        assert len(ranks) == 3

    def test_missing_name_defaults_to_unknown(self):
        data = {"ranks": [{"name": "f1", "expression": "x", "direction": 1}] * 2}
        name, _ = _extract_ranks(data)
        assert name == "unknown"

    def test_raises_on_single_rank(self):
        data = self._make_data(n_ranks=1)
        with pytest.raises(ClickExit):
            _extract_ranks(data)

    def test_raises_on_missing_ranks_key(self):
        with pytest.raises(ClickExit):
            _extract_ranks({"name": "s1"})

    def test_raises_on_empty_ranks(self):
        with pytest.raises(ClickExit):
            _extract_ranks({"name": "s1", "ranks": []})


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


class TestCliSmoke:
    def test_opt_help(self):
        result = runner.invoke(app, ["opt", "--help"])
        assert result.exit_code == 0
        assert "yaml" in result.output.lower()
        assert "n-trials" in result.output.lower()

    def test_opt_missing_yaml_fails(self, tmp_path):
        nonexistent = str(tmp_path / "no_such_file.yaml")
        result = runner.invoke(app, ["opt", "--yaml", nonexistent])
        # 应以非零退出码结束（文件不存在）
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# YAML 写回测试
# ---------------------------------------------------------------------------


class TestYamlWriteback:
    """YAML 写回测试（扁平格式）。"""

    def _make_yaml_content(self) -> str:
        return (
            "name: test_strat\n"
            "type: single\n"
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
        from ruamel.yaml import YAML

        yaml_file = tmp_path / "test_strat.yaml"
        yaml_file.write_text(self._make_yaml_content(), encoding="utf-8")

        # 模拟：将最优权重写回（扁平格式，直接更新根节点 ranks）
        new_weights = np.array([0.3, 0.7])
        _yaml = YAML()
        _yaml.preserve_quotes = True
        with yaml_file.open("r", encoding="utf-8") as f:
            data = _yaml.load(f)

        for rank_def, w in zip(data.get("ranks", []), new_weights):
            rank_def["weight"] = round(float(w), 6)

        with yaml_file.open("w", encoding="utf-8") as f:
            _yaml.dump(data, f)

        # 读回验证
        with yaml_file.open("r", encoding="utf-8") as f:
            result_data = _yaml.load(f)

        written_weights = [r["weight"] for r in result_data["ranks"]]
        assert abs(written_weights[0] - 0.3) < 1e-6
        assert abs(written_weights[1] - 0.7) < 1e-6

    def test_structure_preserved_after_writeback(self, tmp_path):
        from ruamel.yaml import YAML

        yaml_file = tmp_path / "test_strat2.yaml"
        yaml_file.write_text(self._make_yaml_content(), encoding="utf-8")

        _yaml = YAML()
        _yaml.preserve_quotes = True
        with yaml_file.open("r", encoding="utf-8") as f:
            data = _yaml.load(f)

        # 写回不改变 expression 和 direction
        for rank_def in data.get("ranks", []):
            rank_def["weight"] = 0.5

        with yaml_file.open("w", encoding="utf-8") as f:
            _yaml.dump(data, f)

        with yaml_file.open("r", encoding="utf-8") as f:
            result_data = _yaml.load(f)

        ranks = result_data["ranks"]
        assert ranks[0]["expression"] == "CLOSE.shift(1)"
        assert ranks[1]["direction"] == -1
        assert ranks[0]["name"] == "f1"
