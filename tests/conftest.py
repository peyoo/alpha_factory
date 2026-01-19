"""
Pytest 配置与共享 Fixtures。

提供项目范围内的 Mock 数据生成器和通用测试工具。
"""

import pytest
from datetime import datetime, timedelta
from typing import List
import polars as pl
from loguru import logger


# ============================================================================
# Mock 数据生成器
# ============================================================================


@pytest.fixture
def mock_daily_bars():
    """生成模拟日线行情数据"""
    dates = [
        (datetime(2023, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(10)
    ]
    assets = ["000001.SZ", "000002.SZ", "000003.SZ"]

    data = {
        "_DATE_": [],
        "_ASSET_": [],
        "RAW_OPEN": [],
        "RAW_HIGH": [],
        "RAW_LOW": [],
        "RAW_CLOSE": [],
        "OPEN": [],
        "HIGH": [],
        "LOW": [],
        "CLOSE": [],
        "VOLUME": [],
        "AMOUNT": [],
    }

    for date in dates:
        for asset in assets:
            data["_DATE_"].append(date)
            data["_ASSET_"].append(asset)
            # 模拟价格数据
            base_price = 10.0 + hash(asset) % 5
            data["RAW_OPEN"].append(base_price)
            data["RAW_HIGH"].append(base_price + 0.5)
            data["RAW_LOW"].append(base_price - 0.3)
            data["RAW_CLOSE"].append(base_price + 0.2)
            data["OPEN"].append(base_price)  # 后复权价格
            data["HIGH"].append(base_price + 0.5)
            data["LOW"].append(base_price - 0.3)
            data["CLOSE"].append(base_price + 0.2)
            data["VOLUME"].append(1000000.0)  # 成交量（单位：股）
            data["AMOUNT"].append(10000000.0)  # 成交额（单位：元）

    df = pl.DataFrame(data).with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y-%m-%d")
    )
    return df


@pytest.fixture
def mock_daily_basic():
    """生成模拟每日基础指标数据"""
    dates = [
        (datetime(2023, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(10)
    ]
    assets = ["000001.SZ", "000002.SZ", "000003.SZ"]

    data = {
        "_DATE_": [],
        "_ASSET_": [],
        "turnover_rate": [],
        "pe": [],
        "pb": [],
        "total_mv": [],
    }

    for date in dates:
        for asset in assets:
            data["_DATE_"].append(date)
            data["_ASSET_"].append(asset)
            data["turnover_rate"].append(0.02 + hash(asset) % 5 * 0.01)
            data["pe"].append(10.0 + hash(asset) % 10)
            data["pb"].append(1.0 + hash(asset) % 3)
            data["total_mv"].append(100000000.0)

    df = pl.DataFrame(data).with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y-%m-%d")
    )
    return df


@pytest.fixture
def mock_market_status():
    """生成模拟市场状态数据"""
    dates = [
        (datetime(2023, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(10)
    ]
    assets = ["000001.SZ", "000002.SZ", "000003.SZ"]

    data = {
        "_DATE_": [],
        "_ASSET_": [],
        "up_limit": [],
        "down_limit": [],
        "is_st": [],
        "is_suspended": [],
    }

    for date in dates:
        for i, asset in enumerate(assets):
            data["_DATE_"].append(date)
            data["_ASSET_"].append(asset)
            base_price = 10.0 + i * 2
            # 涨跌停价
            data["up_limit"].append(base_price * 1.1)
            data["down_limit"].append(base_price * 0.9)
            # ST 标记（第 2 个股票在第 5 天标记为 ST）
            data["is_st"].append(i == 1 and int(date[-2:]) >= 5)
            # 停牌标记（第 3 个股票在第 7-8 天停牌）
            day = int(date[-2:])
            data["is_suspended"].append(i == 2 and 7 <= day <= 8)

    df = pl.DataFrame(data).with_columns(
        pl.col("_DATE_").str.strptime(pl.Date, "%Y-%m-%d")
    )
    return df


# ============================================================================
# 断言工具
# ============================================================================


def assert_schema_valid(df: pl.DataFrame, expected_schema: dict):
    """验证 DataFrame Schema 是否符合预期"""
    actual_schema = df.schema
    for col_name, expected_type in expected_schema.items():
        assert col_name in actual_schema, f"缺少列: {col_name}"
        assert actual_schema[col_name] == expected_type, (
            f"列 {col_name} 类型不匹配: 期望 {expected_type}, 实际 {actual_schema[col_name]}"
        )
    logger.info(f"✓ Schema 验证通过: {len(expected_schema)} 列符合预期")


def assert_no_future_leakage(df: pl.DataFrame, target_col: str, feature_cols: List[str]):
    """
    验证是否存在未来数据泄露。

    假设: target_col 是下一日标签，feature_cols 是当日特征。
    检查: 最后一行的 target_col 应为空（无下一日数据）。
    """
    last_row_target = df[-1][target_col].item()
    assert last_row_target is None, f"❌ 检测到未来数据泄露: 最后一行的 {target_col} 应为 None"
    logger.info(f"✓ 无未来数据泄露: {target_col} 验证通过")


def assert_null_distribution(df: pl.DataFrame, col: str, max_null_rate: float = 0.05):
    """验证某列的空值分布是否在预期范围内"""
    null_count = df[col].null_count()
    null_rate = null_count / len(df)
    assert null_rate <= max_null_rate, (
        f"❌ {col} 空值率过高: {null_rate:.2%} > {max_null_rate:.2%}"
    )
    logger.info(f"✓ {col} 空值分布正常: {null_rate:.2%}")


# ============================================================================
# 日志配置
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_logger():
    """配置测试环境的日志"""
    logger.remove()  # 移除默认处理器
    logger.add(
        lambda msg: print(msg, end=""),
        format="<level>[{level: <8}]</level> {message}",
        level="INFO",
    )
    logger.info("✓ 测试环境日志已配置")
