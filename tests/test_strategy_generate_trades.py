"""tests/test_strategy_generate_trades.py

单元测试：StrategyConfig.generate_potential_trades
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from alpha_factory.config.strategy import StrategyConfig, FactorRank
from alpha_factory.utils.schema import F


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(
    factor_map: dict[str, list[float]],
    close_map: dict[str, list[float]] | None = None,
    pool_mask_map: dict[str, list[bool]] | None = None,
) -> pl.DataFrame:
    """构建多资产 mock DataFrame。

    factor_map: {asset: [factor_day0, factor_day1, ...]}
    每个日期内各资产 factor 值应唯一，以保证 rank 结果确定。
    ascending=False（默认）时：factor 值越大 → rank 数字越小（越靠前）。

    close_map: {asset: [close_day0, ...]}, 若 None 全部为 100.0
    pool_mask_map: {asset: [bool_day0, ...]}, 若 None 全部为 True
    """
    assets = sorted(factor_map.keys())
    n_days = len(next(iter(factor_map.values())))
    start = datetime.date(2023, 1, 3)
    dates = [start + datetime.timedelta(days=i) for i in range(n_days)]

    rows = []
    for asset in assets:
        for i, date in enumerate(dates):
            factor = factor_map[asset][i]
            close = (close_map or {}).get(asset, [100.0] * n_days)[i]
            mask = (pool_mask_map or {}).get(asset, [True] * n_days)[i]
            rows.append(
                {
                    F.DATE: date,
                    F.ASSET: asset,
                    F.POOL_MASK: mask,
                    "factor": float(factor),
                    F.CLOSE: float(close),
                    F.TOTAL_MV: 1e9,
                    F.PE: 20.0,
                    F.PB: 2.0,
                    F.TURNOVER_RATE: 0.01,
                }
            )

    return pl.DataFrame(rows)


def _make_cfg(buy_rank: int = 1, sell_rank: int = 2) -> StrategyConfig:
    """构造最小 StrategyConfig。"""
    return StrategyConfig(
        name="test",
        pool="main_small_pool",
        ranks=[FactorRank(expression="factor", name="factor")],
        buy_rank=buy_rank,
        sell_rank=sell_rank,
    )


# ---------------------------------------------------------------------------
# 测试：基本买卖触发
# ---------------------------------------------------------------------------


def test_buy_triggered():
    """rank <= buy_rank 时应产生买入记录。

    3 资产/日（ascending=False）：factor 越大 → rank 越小（越靠前）。
    day0: A=10(rank1) ≤ buy_rank=1 → BUY A
    day1: A=1(rank3) > sell_rank=2 → SELL A
    """
    cfg = _make_cfg(buy_rank=1, sell_rank=2)
    df = _make_df({"A": [10.0, 1.0, 5.0], "B": [5.0, 5.0, 5.0], "C": [1.0, 10.0, 1.0]})
    trades = cfg.generate_potential_trades(df, "factor", include_open=False)
    a_trades = trades.filter(pl.col(F.ASSET) == "A")
    assert len(a_trades) == 1
    row = a_trades.row(0, named=True)
    assert row["buy_date"] == datetime.date(2023, 1, 3)
    assert row["sell_date"] == datetime.date(2023, 1, 4)


def test_no_buy_when_rank_too_high():
    """rank 始终 > buy_rank，不应产生任何交易。

    buy_rank=1；A factor 始终最小（rank=3），B rank=2，均 > buy_rank=1。
    C factor 最高（rank=1 ≤ 1）会买入，但从不卖出 → include_open=False → 0 trade。
    """
    cfg = _make_cfg(buy_rank=1, sell_rank=2)
    # C 一直 rank=1 ≤ sell_rank=2，不触发 rank > sell_rank，没有卖出 → 0 完成交易
    df = _make_df({"A": [1.0, 1.0, 1.0], "B": [5.0, 5.0, 5.0], "C": [10.0, 10.0, 10.0]})
    trades = cfg.generate_potential_trades(df, "factor", include_open=False)
    assert len(trades) == 0


# ---------------------------------------------------------------------------
# 测试：pnl_ret
# ---------------------------------------------------------------------------


def test_pnl_ret_calculation():
    """pnl_ret = sell_close / buy_close - 1。

    A day0 rank=1 买入(close=100)，day1 rank=3 > sell_rank=2 卖出(close=120)。
    pnl_ret = 120/100 - 1 = 0.2
    """
    cfg = _make_cfg(buy_rank=1, sell_rank=2)
    df = _make_df(
        {"A": [10.0, 1.0, 5.0], "B": [5.0, 5.0, 5.0], "C": [1.0, 10.0, 1.0]},
        {"A": [100.0, 120.0, 120.0]},
    )
    trades = cfg.generate_potential_trades(df, "factor", include_open=False)
    a_trades = trades.filter(pl.col(F.ASSET) == "A")
    assert len(a_trades) == 1
    pnl = a_trades["pnl_ret"][0]
    assert pnl == pytest.approx(120.0 / 100.0 - 1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 测试：hold_days
# ---------------------------------------------------------------------------


def test_hold_days():
    """hold_days = 卖出日 - 买入日 的日历天数。

    A day0(2023-01-03) 买入，day2(2023-01-05) 卖出 → hold_days=2。
    day0: A=10(rank1) BUY; day1: A=10(rank1) ≤ sell_rank=2 → stay;
    day2: A=1(rank3) > sell_rank=2 → SELL
    """
    cfg = _make_cfg(buy_rank=1, sell_rank=2)
    df = _make_df(
        {"A": [10.0, 10.0, 1.0], "B": [5.0, 5.0, 5.0], "C": [1.0, 1.0, 10.0]},
    )
    trades = cfg.generate_potential_trades(df, "factor", include_open=False)
    a_trades = trades.filter(pl.col(F.ASSET) == "A")
    assert len(a_trades) == 1
    assert a_trades["hold_days"][0] == 2


# ---------------------------------------------------------------------------
# 测试：include_open
# ---------------------------------------------------------------------------


def test_include_open_true():
    """末尾仍持仓时 include_open=True 应保留该记录（sell_date=null）。

    A 始终 factor 最高（rank=1 ≤ sell_rank=2），从不触发卖出。
    """
    cfg = _make_cfg(buy_rank=1, sell_rank=2)
    df = _make_df({"A": [10.0, 10.0, 10.0], "B": [5.0, 5.0, 5.0], "C": [1.0, 1.0, 1.0]})
    trades = cfg.generate_potential_trades(df, "factor", include_open=True)
    a_open = trades.filter(pl.col(F.ASSET) == "A")
    assert len(a_open) == 1
    assert a_open["sell_date"][0] is None
    assert a_open["pnl_ret"][0] is None


def test_include_open_false():
    """include_open=False 时开放持仓不应出现在结果中。"""
    cfg = _make_cfg(buy_rank=1, sell_rank=2)
    df = _make_df({"A": [10.0, 10.0, 10.0], "B": [5.0, 5.0, 5.0], "C": [1.0, 1.0, 1.0]})
    trades = cfg.generate_potential_trades(df, "factor", include_open=False)
    a_trades = trades.filter(pl.col(F.ASSET) == "A")
    assert len(a_trades) == 0


# ---------------------------------------------------------------------------
# 测试：多资产独立状态机
# ---------------------------------------------------------------------------


def test_multiple_assets_independent():
    """不同资产的状态机应独立运行，互不影响。

    3 资产（A,B,C）；buy_rank=1, sell_rank=2：
    - A factor 在 10/1 交替 → rank 在 1/3 交替 → 2 笔完成交易
    - B factor 固定=5 → rank 固定=2 > buy_rank=1 → 0 笔交易
    - C factor 固定=2 → rank 固定=3 > buy_rank=1 → 0 笔交易
    """
    cfg = _make_cfg(buy_rank=1, sell_rank=2)
    df = _make_df(
        {
            "A": [10.0, 1.0, 10.0, 1.0],
            "B": [5.0, 5.0, 5.0, 5.0],
            "C": [2.0, 2.0, 2.0, 2.0],
        }
    )
    trades = cfg.generate_potential_trades(df, "factor", include_open=False)
    a_trades = trades.filter(pl.col(F.ASSET) == "A")
    b_trades = trades.filter(pl.col(F.ASSET) == "B")
    assert len(a_trades) == 2
    assert len(b_trades) == 0


# ---------------------------------------------------------------------------
# 测试：多因子时未指定 factor_col 应报错
# ---------------------------------------------------------------------------


def test_multi_factor_missing_factor_col_raises():
    cfg = StrategyConfig(
        name="multi",
        pool="main_small_pool",
        ranks=[
            FactorRank(expression="f1", name="f1"),
            FactorRank(expression="f2", name="f2"),
        ],
        buy_rank=1,
        sell_rank=2,
    )
    df = _make_df({"A": [10.0, 1.0, 5.0], "B": [5.0, 5.0, 5.0], "C": [1.0, 10.0, 1.0]})
    with pytest.raises(ValueError, match="多因子"):
        cfg.generate_potential_trades(df, factor_col=None)


# ---------------------------------------------------------------------------
# 测试：pool_mask=False 的行不参与排名（rank=999999）
# ---------------------------------------------------------------------------


def test_pool_mask_false_excluded():
    """pool_mask=False 的行 rank 应为 999999，不触发买入。

    A 始终 pool_mask=False → rank=999999 >> buy_rank=1 → 永不买入。
    B,C pool_mask=True，但无足够 factor 变化触发完整买卖 → include_open=False → 0 trade。
    """
    cfg = _make_cfg(buy_rank=1, sell_rank=2)
    n_days = 4
    start = datetime.date(2023, 1, 3)
    dates = [start + datetime.timedelta(days=i) for i in range(n_days)]
    rows = []
    for i, date in enumerate(dates):
        rows.append(
            {
                F.DATE: date,
                F.ASSET: "A",
                F.POOL_MASK: False,
                "factor": 100.0,
                F.CLOSE: 100.0,
                F.TOTAL_MV: 1e9,
                F.PE: 20.0,
                F.PB: 2.0,
                F.TURNOVER_RATE: 0.01,
            }
        )
        rows.append(
            {
                F.DATE: date,
                F.ASSET: "B",
                F.POOL_MASK: True,
                "factor": 5.0,
                F.CLOSE: 100.0,
                F.TOTAL_MV: 1e9,
                F.PE: 20.0,
                F.PB: 2.0,
                F.TURNOVER_RATE: 0.01,
            }
        )
        rows.append(
            {
                F.DATE: date,
                F.ASSET: "C",
                F.POOL_MASK: True,
                "factor": 1.0,
                F.CLOSE: 100.0,
                F.TOTAL_MV: 1e9,
                F.PE: 20.0,
                F.PB: 2.0,
                F.TURNOVER_RATE: 0.01,
            }
        )
    df = pl.DataFrame(rows)
    # A pool_mask=False → rank=999999, 永不满足 rank ≤ buy_rank=1
    trades = cfg.generate_potential_trades(df, "factor", include_open=False)
    a_trades = trades.filter(pl.col(F.ASSET) == "A")
    assert len(a_trades) == 0


# ---------------------------------------------------------------------------
# 测试：新增 SHAP 辅助函数
# ---------------------------------------------------------------------------


def test_detect_shap_threshold_zero_crossing():
    """线性 shap 从 -1 到 +1，零点约在 x=0.5 处。"""
    from alpha_factory.cli._trades_viz import _detect_shap_threshold
    import numpy as np

    x = np.linspace(0.0, 1.0, 100)
    s = np.linspace(-1.0, 1.0, 100)
    thresh = _detect_shap_threshold(x, s)
    assert thresh is not None
    assert abs(thresh - 0.5) < 0.06  # 容许 6% 误差（桶间线性插值精度）


def test_shap_dependence_grid_returns_one_per_feature():
    """_shap_dependence_grid 应为每个特征返回一个非空 base64 字符串。"""
    from alpha_factory.cli._trades_viz import _shap_dependence_grid
    import numpy as np

    rng = np.random.default_rng(0)
    n_feat = 3
    n_samples = 60
    shap_vals = rng.standard_normal((n_samples, n_feat))
    X = rng.standard_normal((n_samples, n_feat))
    feature_names = ["f1", "f2", "f3"]

    result = _shap_dependence_grid(shap_vals, X, feature_names)
    assert len(result) == n_feat
    assert all(isinstance(s, str) and len(s) > 10 for s in result)
