"""_loader.py — cli 层共享的 pool 解析 + DataProvider 加载工具。

替代 opt.py / opt2.py / eval_core.py / trades.py 中重复的
_resolve_pool / _get_pool_universe 逻辑。
"""

from __future__ import annotations

from typing import Optional

import polars as pl
from loguru import logger

from alpha_factory.cli.utils import PoolUniverseEnum
from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.data_provider.pool import PoolUniverse


def resolve_pool(pool_name: str) -> PoolUniverse:
    """将股票池名称字符串解析为 PoolUniverse 实例。

    遍历 PoolUniverseEnum，找到 name 匹配的成员并实例化返回。
    若未找到则抛出 ValueError（包含所有可选值）。
    """
    for member in PoolUniverseEnum:
        instance = member.value()
        if instance.name == pool_name:
            return instance
    valid = ", ".join(m.value().name for m in PoolUniverseEnum)
    raise ValueError(f"未知股票池 {pool_name!r}，可选值: {valid}")


def load_pool_lf(
    pool: PoolUniverse,
    exprs: list[str],
    start_date: str,
    end_date: Optional[str],
    actions: Optional[list] = None,
) -> pl.LazyFrame:
    """初始化 DataProvider 并加载股票池数据，返回 LazyFrame。

    Args:
        pool: PoolUniverse 实例（由 resolve_pool 获取）
        exprs: 因子表达式列表，格式 "name=expression"
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD（None 表示至最新）
        actions: 可选的预处理 action 列表（来自 StrategyConfig.build_actions()）

    Returns:
        pl.LazyFrame: 包含因子列和标签列的惰性数据框
    """
    logger.info(
        f"加载数据: pool={pool.name}, start={start_date}, end={end_date}, "
        f"factors={len(exprs)}"
    )
    dp = DataProvider()
    kwargs: dict = {"exprs": exprs}
    if actions is not None:
        kwargs["actions"] = actions
    return dp.load_pool_data(pool, start_date, end_date, **kwargs)
