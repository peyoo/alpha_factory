"""
tests/test_backtest_consistency.py — 回测引擎一致性测试

验证 Numba JIT 实现 (backtest_quick_daily/_core_evolution_engine)
与 Python 参考实现 (backtest_daily_evolving) 的输出是否一致。
"""

from __future__ import annotations

from typing import Dict

import polars as pl
import pytest

from alpha_factory.evaluation.backtest.daily_evolving import backtest_daily_evolving
from alpha_factory.evaluation.backtest.quick_daily import backtest_quick_daily
from alpha_factory.utils.schema import F


def _make_synthetic_data(
    n_dates: int = 5,
    n_assets: int = 6,
    seed: int = 42,
) -> pl.DataFrame:
    """生成用于回测一致性测试的合成数据。"""
    import datetime

    rng = __import__("numpy").random.default_rng(seed)

    base = datetime.date(2024, 1, 2)
    rows = []
    for d in range(n_dates):
        date = base + datetime.timedelta(days=d)
        # 跳过周末
        while date.weekday() >= 5:
            date += datetime.timedelta(days=1)
        for a in range(n_assets):
            asset = f"{100000 + a:06d}.SZ"
            rows.append(
                {
                    F.DATE: date,
                    F.ASSET: asset,
                    "factor": float(rng.uniform(-1, 1)),
                    F.CLOSE: float(rng.uniform(5, 30)),
                    F.VWAP: float(rng.uniform(5, 30)),
                    F.POOL_MASK: True,
                    F.IS_SUSPENDED: False,
                    F.IS_UP_LIMIT: False,
                    F.IS_DOWN_LIMIT: False,
                }
            )
    return pl.DataFrame(rows)


def _run_both_engines(df: pl.DataFrame, **kwargs) -> tuple[Dict, Dict]:
    """用同一份数据分别调用两个引擎，返回 (python_result, numba_result)。"""
    py_result = backtest_daily_evolving(df, **kwargs)
    nb_result = backtest_quick_daily(df, **kwargs)
    return py_result, nb_result


# ── 基本一致性测试 ──────────────────────────────────────────────────────


def test_daily_results_match_default_params():
    """默认参数下，两个引擎的 daily_results 应完全一致。"""
    df = _make_synthetic_data(n_dates=10, n_assets=8, seed=42)
    py_result, nb_result = _run_both_engines(
        df,
        factor_col="factor",
        n_buy=3,
        sell_rank=6,
        cost_rate=0.003,
        exec_price=F.VWAP,
    )

    py_daily = py_result["daily_results"].sort(F.DATE)
    nb_daily = nb_result["daily_results"].sort(F.DATE)

    for col in ["NAV", "NET_RET", "TURNOVER", "COUNT", "RAW_RET"]:
        assert py_daily[col].to_list() == pytest.approx(
            nb_daily[col].to_list(), abs=1e-10
        ), f"{col} 不一致"


def test_trade_details_stats_match():
    """交易明细的汇总统计应匹配（排序不影响汇总）。"""
    df = _make_synthetic_data(n_dates=20, n_assets=10, seed=99)
    py_result, nb_result = _run_both_engines(
        df,
        factor_col="factor",
        n_buy=4,
        sell_rank=8,
        cost_rate=0.003,
        exec_price=F.VWAP,
    )

    py_trades = py_result["trade_details"]
    nb_trades = nb_result["trade_details"]

    # 交易笔数一致
    assert len(py_trades) == len(nb_trades), (
        f"交易笔数不同: {len(py_trades)} vs {len(nb_trades)}"
    )

    # pnl_ret 排序后一致（交易顺序可能不同）
    py_pnl = py_trades["pnl_ret"].sort().to_list()
    nb_pnl = nb_trades["pnl_ret"].sort().to_list()
    assert py_pnl == pytest.approx(nb_pnl, abs=1e-10), "pnl_ret 分布不一致"

    # holding_periods 排序后一致
    py_hp = py_trades["holding_periods"].sort().to_list()
    nb_hp = nb_trades["holding_periods"].sort().to_list()
    assert py_hp == nb_hp, "holding_periods 分布不一致"


# ── 边界条件 ────────────────────────────────────────────────────────────


def test_partial_pool_mask():
    """部分标的被池外标记（POOL_MASK=False），双引擎过滤逻辑一致。"""
    df = _make_synthetic_data(n_dates=10, n_assets=8, seed=7)
    # 将一半标的标记为池外
    rng = __import__("numpy").random.default_rng(789)
    n = len(df)
    mask_out = rng.choice(n, size=int(n * 0.5), replace=False)
    pool_mask = pl.Series([i not in mask_out for i in range(n)])
    df = df.with_columns(pool_mask.alias(F.POOL_MASK))

    py_result, nb_result = _run_both_engines(
        df, factor_col="factor", n_buy=2, sell_rank=4
    )
    py_nav = py_result["daily_results"]["NAV"].to_list()
    nb_nav = nb_result["daily_results"]["NAV"].to_list()
    assert py_nav == pytest.approx(nb_nav, abs=1e-10)


def test_some_suspended():
    """部分标的部分日期停牌，双引擎仍应一致。"""
    df = _make_synthetic_data(n_dates=15, n_assets=8, seed=21)
    rng = __import__("numpy").random.default_rng(123)
    # 随机设置约 15% 的交易日为停牌
    n = len(df)
    suspend_idx = rng.choice(n, size=int(n * 0.15), replace=False)
    mask = pl.Series([i in suspend_idx for i in range(n)])
    df = df.with_columns(
        pl.when(mask).then(True).otherwise(pl.col(F.IS_SUSPENDED)).alias(F.IS_SUSPENDED)
    )

    py_result, nb_result = _run_both_engines(
        df, factor_col="factor", n_buy=3, sell_rank=5
    )

    py_nav = py_result["daily_results"]["NAV"].to_list()
    nb_nav = nb_result["daily_results"]["NAV"].to_list()
    assert py_nav == pytest.approx(nb_nav, abs=1e-10)


def test_some_limit():
    """部分标的部分日期涨停/跌停，双引擎应一致。"""
    df = _make_synthetic_data(n_dates=12, n_assets=8, seed=55)
    rng = __import__("numpy").random.default_rng(456)
    n = len(df)
    # 涨停
    up_idx = rng.choice(n, size=int(n * 0.1), replace=False)
    mask_up = pl.Series([i in up_idx for i in range(n)])
    df = df.with_columns(
        pl.when(mask_up)
        .then(True)
        .otherwise(pl.col(F.IS_UP_LIMIT))
        .alias(F.IS_UP_LIMIT)
    )

    py_result, nb_result = _run_both_engines(
        df, factor_col="factor", n_buy=3, sell_rank=6
    )
    py_nav = py_result["daily_results"]["NAV"].to_list()
    nb_nav = nb_result["daily_results"]["NAV"].to_list()
    assert py_nav == pytest.approx(nb_nav, abs=1e-10)


def test_single_date():
    """只有 1 个交易日时，两个引擎都应稳定运行。"""
    df = _make_synthetic_data(n_dates=1, n_assets=10, seed=1)
    py_result, nb_result = _run_both_engines(
        df, factor_col="factor", n_buy=3, sell_rank=5
    )
    assert len(py_result["daily_results"]) == 1
    assert len(nb_result["daily_results"]) == 1
    assert py_result["daily_results"]["NAV"][0] == pytest.approx(
        nb_result["daily_results"]["NAV"][0], abs=1e-10
    )


def test_two_dates():
    """只有 2 个交易日（T+1 生效一次），双引擎应一致。"""
    df = _make_synthetic_data(n_dates=2, n_assets=10, seed=2)
    py_result, nb_result = _run_both_engines(
        df, factor_col="factor", n_buy=3, sell_rank=5
    )
    py_nav = py_result["daily_results"]["NAV"].to_list()
    nb_nav = nb_result["daily_results"]["NAV"].to_list()
    assert py_nav == pytest.approx(nb_nav, abs=1e-10)


def test_high_turnover_params():
    """高频换手参数（小 n_buy + 低 sell_rank），双引擎一致性。"""
    df = _make_synthetic_data(n_dates=20, n_assets=12, seed=77)
    py_result, nb_result = _run_both_engines(
        df,
        factor_col="factor",
        n_buy=10,
        sell_rank=5,  # sell_rank < n_buy → 高换手
        cost_rate=0.003,
        exec_price=F.VWAP,
    )

    py_nav = py_result["daily_results"]["NAV"].to_list()
    nb_nav = nb_result["daily_results"]["NAV"].to_list()
    assert py_nav == pytest.approx(nb_nav, abs=1e-10)


def test_ascending_true():
    """ascending=True（rank 越小越好），双引擎一致性。"""
    df = _make_synthetic_data(n_dates=15, n_assets=10, seed=88)
    py_result, nb_result = _run_both_engines(
        df,
        factor_col="factor",
        n_buy=3,
        sell_rank=6,
        cost_rate=0.003,
        ascending=True,
    )
    py_nav = py_result["daily_results"]["NAV"].to_list()
    nb_nav = nb_result["daily_results"]["NAV"].to_list()
    assert py_nav == pytest.approx(nb_nav, abs=1e-10)


def test_zero_cost():
    """零费率，双引擎一致性。"""
    df = _make_synthetic_data(n_dates=10, n_assets=8, seed=33)
    py_result, nb_result = _run_both_engines(
        df,
        factor_col="factor",
        n_buy=3,
        sell_rank=6,
        cost_rate=0.0,
    )
    py_nav = py_result["daily_results"]["NAV"].to_list()
    nb_nav = nb_result["daily_results"]["NAV"].to_list()
    assert py_nav == pytest.approx(nb_nav, abs=1e-10)


# ── 大随机测试 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [1, 42, 99, 123, 456])
def test_multiple_random_seeds(seed: int):
    """多组随机种子验证一致性。"""
    df = _make_synthetic_data(n_dates=20, n_assets=12, seed=seed)
    py_result, nb_result = _run_both_engines(
        df,
        factor_col="factor",
        n_buy=4,
        sell_rank=8,
        cost_rate=0.002,
        exec_price=F.VWAP,
    )
    py_nav = py_result["daily_results"]["NAV"].to_list()
    nb_nav = nb_result["daily_results"]["NAV"].to_list()
    assert py_nav == pytest.approx(nb_nav, abs=1e-10), f"seed={seed} NAV 不一致"
