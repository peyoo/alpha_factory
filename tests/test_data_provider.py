"""
数据接入层单元测试。

覆盖模块：
- alpha.data_provider.schema
- alpha.data_provider.cleaner
- alpha.data_provider.reader
- (tushare_source 需要集成测试或 Mock API)

测试覆盖率目标：> 90%
"""

import pytest
import polars as pl
import pandas as pd
from datetime import datetime
from loguru import logger

from alpha.data_provider.schema import (
    get_schema_by_type,
    validate_schema,
    convert_volume_unit,
    convert_amount_unit,
)
from alpha.data_provider.cleaner import (
    clean_daily_bars,
    clean_calendar,
    clean_adj_factors,
    clean_daily_basic,
    clean_market_status,
)
from alpha.data_provider.reader import (
    DataProvider,
)


# ============================================================================
# Schema 模块测试
# ============================================================================


class TestSchema:
    """测试 schema.py 模块"""

    def test_get_schema_by_type_daily_bars(self):
        """验证日线行情 Schema 获取"""
        schema = get_schema_by_type("daily_bars")
        assert schema is not None, "❌ 未能获取日线行情 Schema"
        assert "_DATE_" in schema, "❌ 缺少 _DATE_ 字段"
        assert "_ASSET_" in schema, "❌ 缺少 _ASSET_ 字段"
        assert schema["_DATE_"] == pl.Date, "❌ _DATE_ 类型错误"
        logger.info("✓ 日线行情 Schema 验证通过")

    def test_get_schema_by_type_daily_basic(self):
        """验证每日基础指标 Schema 获取"""
        schema = get_schema_by_type("daily_basic")
        assert schema is not None, "❌ 未能获取每日基础指标 Schema"
        assert "pe" in schema, "❌ 缺少 pe 字段"
        assert "pb" in schema, "❌ 缺少 pb 字段"
        logger.info("✓ 每日基础指标 Schema 验证通过")

    def test_validate_schema_success(self):
        """验证正确的 Schema 验证"""
        df = pl.DataFrame({
            "_DATE_": [datetime(2023, 1, 1)],
            "_ASSET_": ["000001.SZ"],
            "CLOSE": [10.0],
        }).with_columns(pl.col("_DATE_").cast(pl.Date))

        expected_schema = {
            "_DATE_": pl.Date,
            "_ASSET_": pl.String,
            "CLOSE": pl.Float64,
        }

        result = validate_schema(df, expected_schema)
        assert result is True, "❌ Schema 验证失败"
        logger.info("✓ Schema 验证成功")

    def test_validate_schema_missing_column(self):
        """验证缺失列时的错误处理"""
        df = pl.DataFrame({
            "_DATE_": [datetime(2023, 1, 1)],
            "_ASSET_": ["000001.SZ"],
        }).with_columns(pl.col("_DATE_").cast(pl.Date))

        expected_schema = {
            "_DATE_": pl.Date,
            "_ASSET_": pl.String,
            "CLOSE": pl.Float32,  # 这列不存在
        }

        with pytest.raises(AssertionError):
            validate_schema(df, expected_schema)
        logger.info("✓ 缺失列检测验证通过")

    def test_convert_volume_unit(self):
        """验证成交量单位转换（手 -> 股）"""
        lf = pl.LazyFrame({
            "VOLUME": [100.0, 200.0, 300.0]
        })

        result = convert_volume_unit(lf).collect()
        assert result["VOLUME"][0] == 10000.0, "❌ 成交量转换错误"
        assert result["VOLUME"][1] == 20000.0, "❌ 成交量转换错误"
        logger.info("✓ 成交量单位转换验证通过")

    def test_convert_amount_unit(self):
        """验证成交额单位转换（千元 -> 元）"""
        lf = pl.LazyFrame({
            "AMOUNT": [100.0, 200.0, 300.0]
        })

        result = convert_amount_unit(lf).collect()
        assert result["AMOUNT"][0] == 100000.0, "❌ 成交额转换错误"
        assert result["AMOUNT"][1] == 200000.0, "❌ 成交额转换错误"
        logger.info("✓ 成交额单位转换验证通过")


# ============================================================================
# Cleaner 模块测试
# ============================================================================


class TestCleaner:
    """测试 cleaner.py 模块"""

    def test_clean_daily_bars_basic(self, mock_daily_bars):
        """验证日线行情清洗基础功能"""
        df_pandas = mock_daily_bars.to_pandas()

        result = clean_daily_bars(
            df_pandas,
            with_adjustment=False,
            adj_factor_df=None
        )

        result_collected = result.collect()
        assert result_collected.height == df_pandas.shape[0], "❌ 行数不匹配"
        assert "_DATE_" in result_collected.columns, "❌ 缺少 _DATE_ 列"
        assert "_ASSET_" in result_collected.columns, "❌ 缺少 _ASSET_ 列"
        assert "CLOSE" in result_collected.columns, "❌ 缺少 CLOSE 列"
        logger.info(f"✓ 日线行情清洗基础验证通过 (shape={result_collected.shape})")

    def test_clean_daily_bars_schema_validation(self, mock_daily_bars):
        """验证日线行情清洗后的 Schema"""
        df_pandas = mock_daily_bars.to_pandas()

        result = clean_daily_bars(
            df_pandas,
            with_adjustment=False,
            adj_factor_df=None
        )

        result_collected = result.collect()
        # 验证关键列的类型
        assert result_collected["_DATE_"].dtype == pl.Date, "❌ _DATE_ 类型错误"
        assert result_collected["_ASSET_"].dtype == pl.String, "❌ _ASSET_ 类型错误"
        assert result_collected["CLOSE"].dtype == pl.Float32, "❌ CLOSE 类型错误"
        logger.info("✓ 日线行情 Schema 验证通过")

    def test_clean_daily_bars_null_handling(self):
        """验证日线行情中的空值处理"""
        df_pandas = pd.DataFrame({
            "trade_date": ["20230101", "20230102", "20230103"],
            "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [10.0, 11.0, 12.0],
            "high": [10.5, 11.5, 12.5],
            "low": [9.5, 10.5, 11.5],
            "close": [10.2, 11.2, 12.2],
            "vol": [100000.0, 200000.0, None],  # 停牌数据
            "amount": [1000000.0, 2000000.0, 3000000.0],
        })

        result = clean_daily_bars(
            df_pandas,
            with_adjustment=False,
            adj_factor_df=None
        )

        result_collected = result.collect()
        # 验证空值保留
        assert result_collected["VOLUME"].null_count() == 1, "❌ 空值未正确保留"
        logger.info("✓ 空值处理验证通过")

    def test_clean_calendar_basic(self):
        """验证交易日历清洗"""
        df_pandas = pd.DataFrame({
            "cal_date": ["20230101", "20230102", "20230103"],
            "is_open": [0, 1, 1],
        })

        result = clean_calendar(df_pandas)
        result_collected = result.collect()

        assert result_collected.height == 3, "❌ 行数不匹配"
        assert result_collected["_DATE_"].dtype == pl.Date, "❌ _DATE_ 类型错误"
        assert result_collected["is_open"].dtype == pl.Boolean, "❌ is_open 类型错误"
        logger.info("✓ 交易日历清洗验证通过")

    def test_clean_adj_factors_basic(self):
        """验证复权因子清洗"""
        df_pandas = pd.DataFrame({
            "trade_date": ["20230101", "20230102"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "adj_factor": [1.0, 1.0],
        })

        result = clean_adj_factors(df_pandas)
        result_collected = result.collect()

        assert result_collected.height == 2, "❌ 行数不匹配"
        assert result_collected["adj_factor"].dtype == pl.Float32, "❌ adj_factor 类型错误"
        logger.info("✓ 复权因子清洗验证通过")

    def test_clean_daily_basic_basic(self, mock_daily_basic):
        """验证每日基础指标清洗"""
        df_pandas = mock_daily_basic.to_pandas()

        result = clean_daily_basic(df_pandas)
        result_collected = result.collect()

        assert result_collected.height == df_pandas.shape[0], "❌ 行数不匹配"
        assert "pe" in result_collected.columns, "❌ 缺少 pe 列"
        assert "pb" in result_collected.columns, "❌ 缺少 pb 列"
        logger.info("✓ 每日基础指标清洗验证通过")

    def test_clean_market_status_with_st(self):
        """验证 ST 状态标记"""
        df_st = pd.DataFrame({
            "trade_date": ["20230105"],
            "ts_code": ["000002.SZ"],
        })

        result = clean_market_status(stock_st_df=df_st)
        result_collected = result.collect()

        assert "is_st" in result_collected.columns, "❌ 缺少 is_st 列"
        st_rows = result_collected.filter(pl.col("is_st"))
        assert len(st_rows) > 0, "❌ 未正确标记 ST"
        logger.info("✓ ST 状态标记验证通过")

    def test_clean_market_status_with_suspend(self):
        """验证停牌状态标记"""
        df_suspend = pd.DataFrame({
            "suspend_date": ["20230107"],
            "ts_code": ["000003.SZ"],
            "suspend_type": ["停牌"],
        })

        result = clean_market_status(suspend_d_df=df_suspend)
        result_collected = result.collect()

        assert "is_suspended" in result_collected.columns, "❌ 缺少 is_suspended 列"
        logger.info("✓ 停牌状态标记验证通过")


# ============================================================================
# Reader 模块测试
# ============================================================================


class TestDataProvider:
    """测试 reader.py 模块"""

    def test_data_provider_initialization(self):
        """验证 DataProvider 初始化"""
        provider = DataProvider()
        assert provider is not None, "❌ DataProvider 初始化失败"
        logger.info("✓ DataProvider 初始化验证通过")

    def test_parse_date_valid(self):
        """验证日期解析（有效格式）"""
        date_str = "20230101"
        result = DataProvider._parse_date(date_str)

        assert result.year == 2023, "❌ 年份解析错误"
        assert result.month == 1, "❌ 月份解析错误"
        assert result.day == 1, "❌ 日期解析错误"
        logger.info("✓ 日期解析验证通过")

    def test_parse_date_invalid(self):
        """验证日期解析（无效格式）"""
        with pytest.raises(ValueError):
            DataProvider._parse_date("2023-01-01")  # 错误的格式
        logger.info("✓ 无效日期检测验证通过")

    def test_estimate_row_count(self):
        """验证行数估计"""
        start_date = "20230101"
        end_date = "20231231"

        count = DataProvider._estimate_row_count(start_date, end_date)
        assert count > 0, "❌ 行数估计为零"
        assert isinstance(count, int), "❌ 行数估计类型错误"
        logger.info(f"✓ 行数估计验证通过 (预期 {count} 行)")

    def test_should_load_table_empty_columns(self):
        """验证表加载决策（无列指定时全部加载）"""
        result = DataProvider._should_load_table("daily_basic", set())
        assert result is True, "❌ 应加载所有表"
        logger.info("✓ 表加载决策验证通过")

    def test_should_load_table_with_specific_columns(self):
        """验证表加载决策（指定列时有条件加载）"""
        # 请求包含 PE 列，应加载 daily_basic 表
        result = DataProvider._should_load_table("daily_basic", {"pe"})
        assert result is True, "❌ 应加载 daily_basic 表"

        # 请求不包含 daily_basic 的列，不加载
        result = DataProvider._should_load_table("daily_basic", {"CLOSE"})
        assert result is False, "❌ 不应加载 daily_basic 表"
        logger.info("✓ 条件表加载决策验证通过")


# ============================================================================
# 集成测试
# ============================================================================


class TestDataProviderIntegration:
    """数据接入层集成测试"""

    def test_schema_cleaner_integration(self, mock_daily_bars):
        """验证 Schema 与 Cleaner 的集成"""
        df_pandas = mock_daily_bars.to_pandas()

        # 清洗数据
        result_lf = clean_daily_bars(df_pandas, with_adjustment=False)
        result = result_lf.collect()

        # 验证 Schema
        from alpha.data_provider.schema import DAILY_BARS_SCHEMA
        validate_schema(result, DAILY_BARS_SCHEMA)

        logger.info("✓ Schema 与 Cleaner 集成验证通过")

    def test_multiple_cleaners_output_schema(self, mock_daily_basic):
        """验证多个 Cleaner 的输出 Schema 一致性"""
        df_pandas = mock_daily_basic.to_pandas()

        result = clean_daily_basic(df_pandas)
        result_collected = result.collect()

        from alpha.data_provider.schema import DAILY_BASIC_SCHEMA
        validate_schema(result_collected, DAILY_BASIC_SCHEMA)
        logger.info("✓ 多 Cleaner 输出 Schema 验证通过")


# ============================================================================
# 边缘情况测试
# ============================================================================


class TestEdgeCases:
    """边缘情况与量化特定场景测试"""

    def test_empty_dataframe_handling(self):
        """验证空数据处理"""
        df_pandas = pd.DataFrame({
            "trade_date": [],
            "ts_code": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "vol": [],
            "amount": [],
        })

        # 应该正常处理（返回空 LazyFrame）
        result = clean_daily_bars(df_pandas, with_adjustment=False)
        result_collected = result.collect()
        assert result_collected.height == 0, "❌ 空数据处理失败"
        logger.info("✓ 空数据处理验证通过")

    def test_all_null_values(self):
        """验证全空值列的处理"""
        df_pandas = pd.DataFrame({
            "trade_date": ["20230101", "20230102"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "open": [None, None],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "vol": [1000000.0, 2000000.0],
            "amount": [10000000.0, 20000000.0],
        })

        result = clean_daily_bars(df_pandas, with_adjustment=False)
        result_collected = result.collect()

        # OPEN 列应保留全空值
        assert result_collected["OPEN"].null_count() == 2, "❌ 全空值处理不当"
        logger.info("✓ 全空值处理验证通过")

    def test_extreme_prices(self):
        """验证极限价格处理"""
        df_pandas = pd.DataFrame({
            "trade_date": ["20230101"],
            "ts_code": ["000001.SZ"],
            "open": [999999.99],
            "high": [1000000.0],
            "low": [999999.0],
            "close": [999999.50],
            "vol": [1.0],  # 极小成交量
            "amount": [1000000.0],
        })

        result = clean_daily_bars(df_pandas, with_adjustment=False)
        result_collected = result.collect()

        # 验证极限价格未被截断
        assert result_collected["HIGH"][0] == 1000000.0, "❌ 极限价格处理错误"
        logger.info("✓ 极限价格处理验证通过")

    def test_duplicate_dates_and_assets(self):
        """验证重复日期和资产的处理"""
        df_pandas = pd.DataFrame({
            "trade_date": ["20230101", "20230101"],  # 重复日期
            "ts_code": ["000001.SZ", "000001.SZ"],   # 重复资产
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "vol": [1000000.0, 2000000.0],
            "amount": [10000000.0, 20000000.0],
        })

        result = clean_daily_bars(df_pandas, with_adjustment=False)
        result_collected = result.collect()

        # 应保留两行（数据清洗不进行去重）
        assert result_collected.height == 2, "❌ 重复数据处理不当"
        logger.info("✓ 重复数据处理验证通过")


# ============================================================================
# 测试总结
# ============================================================================

if __name__ == "__main__":
    # 运行全部测试
    # pytest tests/test_data_provider.py -v --cov=alpha.data_provider
    logger.info("✓ 所有数据接入层测试已定义")
