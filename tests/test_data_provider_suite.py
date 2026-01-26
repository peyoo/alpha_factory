"""
Data Provider 模块完整测试套件

测试覆盖：
1. TushareDataService - 数据同步服务
2. HDF5CacheManager - 缓存管理
3. DataProvider - 数据读取接口
4. TradeCalendarManager - 交易日历
5. StockAssetsManager - 资产管理
6. UnifiedFactorBuilder - 因子构建
"""

import pytest
import pandas as pd
import polars as pl
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from alpha.data_provider.tushare_service import TushareDataService, RateLimiter
from alpha.data_provider.cache_manager import HDF5CacheManager
from alpha.data_provider.data_provider import DataProvider
from alpha.data_provider.trade_calendar_manager import TradeCalendarManager
from alpha.data_provider.stock_assets_manager import StockAssetsManager
from alpha.utils.config import settings


# ============================================================================
# RateLimiter 测试
# ============================================================================

class TestRateLimiter:
    """限流器测试"""

    def test_vip_interval(self):
        """VIP 账户间隔时间"""
        rl = RateLimiter(is_vip=True)
        assert abs(rl.min_interval - 0.075) < 0.001

    def test_free_interval(self):
        """免费账户间隔时间"""
        rl = RateLimiter(is_vip=False)
        assert abs(rl.min_interval - 0.3) < 0.001

    def test_wait_logic(self):
        """等待逻辑"""
        rl = RateLimiter(is_vip=True)
        import time

        t1 = time.time()
        rl.wait()
        rl.wait()
        elapsed = time.time() - t1

        # 第二次 wait 应该会睡眠，总时间 >= 75ms
        assert elapsed >= 0.07


# ============================================================================
# HDF5CacheManager 测试
# ============================================================================

class TestHDF5CacheManager:
    """HDF5 缓存管理器测试"""

    @pytest.fixture
    def cache_manager(self):
        """创建临时缓存管理器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield HDF5CacheManager(Path(tmpdir))

    def test_initialization(self, cache_manager):
        """缓存管理器初始化"""
        assert cache_manager.cache_dir.exists()

    def test_save_and_is_cached(self, cache_manager):
        """保存和检查缓存"""
        df = pd.DataFrame({
            "trade_date": ["2023-01-01"],
            "ts_code": ["000001.SZ"],
            "close": [10.0]
        })

        cache_manager.save_to_hdf5("daily", "20230101", df)
        assert cache_manager.is_cached("daily", "20230101")
        assert not cache_manager.is_cached("daily", "20230102")

    def test_load_from_hdf5(self, cache_manager):
        """从 HDF5 加载数据"""
        df = pd.DataFrame({
            "trade_date": ["2023-01-01", "2023-01-02"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.5]
        })

        cache_manager.save_to_hdf5("daily", "20230101", df.iloc[:1])
        cache_manager.save_to_hdf5("daily", "20230102", df.iloc[1:])

        result = cache_manager.load_from_hdf5("daily", ["20230101", "20230102"])
        assert result is not None
        assert len(result) == 2

    def test_clear_cache(self, cache_manager):
        """清理缓存"""
        df = pd.DataFrame({
            "trade_date": ["2023-01-01"],
            "ts_code": ["000001.SZ"],
            "close": [10.0]
        })

        cache_manager.save_to_hdf5("daily", "20230101", df)
        assert cache_manager.is_cached("daily", "20230101")

        cache_manager.clear_cache("daily")
        assert not cache_manager.is_cached("daily", "20230101")


# ============================================================================
# DataProvider 测试
# ============================================================================

class TestDataProvider:
    """数据读取接口测试"""

    @pytest.fixture
    def provider(self):
        """创建数据提供者"""
        yield DataProvider()

    def test_initialization(self, provider):
        """初始化"""
        assert provider.warehouse_dir is not None
        assert provider.factor_dir is not None

    def test_load_data_date_parsing(self, provider):
        """日期解析"""
        with pytest.raises(ValueError):
            # 错误的日期格式应该抛出异常
            provider.load_data("2023-01-01", "20230131")

    def test_validate_schema_success(self):
        """Schema 验证成功"""
        provider = DataProvider()

        lf = pl.LazyFrame({
            "_DATE_": [date(2023, 1, 1)],
            "_ASSET_": ["000001.SZ"],
            "CLOSE": [10.0],
            "VOLUME": [1000.0],
            "IS_SUSPENDED": [False],
        })

        result = provider.validate_schema(lf)
        assert result is True

    def test_validate_schema_missing_column(self):
        """Schema 缺少必需列"""
        provider = DataProvider()

        lf = pl.LazyFrame({
            "_DATE_": [date(2023, 1, 1)],
            "_ASSET_": ["000001.SZ"],
            "CLOSE": [10.0],
            # 缺少 IS_SUSPENDED
            "VOLUME": [1000.0],
        })

        result = provider.validate_schema(lf)
        assert result is False

    def test_validate_schema_wrong_type(self):
        """Schema 类型错误"""
        provider = DataProvider()

        lf = pl.LazyFrame({
            "_DATE_": ["2023-01-01"],  # ❌ 应该是 Date 而非 String
            "_ASSET_": ["000001.SZ"],
            "CLOSE": [10.0],
            "IS_SUSPENDED": [False],
            "VOLUME": [1000.0],
        })

        result = provider.validate_schema(lf)
        assert result is False


# ============================================================================
# TradeCalendarManager 测试
# ============================================================================

class TestTradeCalendarManager:
    """交易日历管理器测试"""

    @pytest.fixture
    def manager(self):
        """创建临时交易日历管理器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield TradeCalendarManager(Path(tmpdir) / "trade_calendar.parquet")

    def test_initialization(self, manager):
        """初始化"""
        assert manager.path is not None

    def test_is_trade_day(self, manager):
        """判断是否为交易日"""
        # 没有数据时应该返回 False
        assert manager.is_trade_day(date(2023, 1, 2)) is False

    def test_offset_basic(self, manager):
        """基本的偏移计算"""
        # 没有数据时应该返回 None
        result = manager.offset(date(2023, 1, 2), 1)
        assert result is None

    def test_get_trade_days_empty(self, manager):
        """获取空的交易日列表"""
        result = manager.get_trade_days(date(2023, 1, 1), date(2023, 1, 31))
        assert result is not None


# ============================================================================
# StockAssetsManager 测试
# ============================================================================

class TestStockAssetsManager:
    """资产管理器测试"""

    @pytest.fixture
    def manager(self):
        """创建临时资产管理器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield StockAssetsManager(Path(tmpdir) / "stock_assets.parquet")

    def test_initialization(self, manager):
        """初始化"""
        assert manager._df is not None
        assert manager.stock_type is not None

    def test_get_asset_mapping_empty(self, manager):
        """获取空的资产映射"""
        mapping = manager.get_asset_mapping()
        assert isinstance(mapping, dict)

    def test_update_assets(self, manager):
        """更新资产"""
        snap = pl.DataFrame({
            "asset": ["000001.SZ", "000002.SZ"],
            "name": ["平安", "万科"],
            "list_date": [date(2020, 1, 1), date(2020, 1, 2)],
            "delist_date": [None, None],
            "exchange": ["SSE", "SSE"],
        }).cast(manager.schema)

        manager.update_assets(snap)
        assert manager._df.height >= 2

    def test_get_properties(self, manager):
        """获取资产属性"""
        props = manager.get_properties()
        assert isinstance(props, pl.DataFrame)


# ============================================================================
# TushareDataService 测试（带 Mock）
# ============================================================================

class TestTushareDataService:
    """Tushare 数据服务测试（Mock 模式）"""

    @pytest.fixture
    def mock_service(self):
        """创建 Mock Tushare 服务"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(vars(settings), {
                'TUSHARE_TOKEN': 'mock_token',
                'RAW_DATA_DIR': Path(tmpdir) / 'raw',
                'WAREHOUSE_DIR': Path(tmpdir) / 'warehouse',
                'is_vip': True,
            }):
                with patch('alpha.data_provider.tushare_service.TushareDataService._init_tushare') as mock_init:
                    mock_pro = MagicMock()
                    mock_init.return_value = mock_pro

                    service = TushareDataService()
                    service.pro = mock_pro

                    yield service

    def test_initialization(self, mock_service):
        """初始化"""
        assert mock_service.token is not None
        assert mock_service.rate_limiter is not None
        assert mock_service.cache_manager is not None

    def test_process_raw_df_date_normalization(self, mock_service):
        """日期规范化"""
        df = pd.DataFrame({
            "trade_date": ["20230101"],
            "ts_code": ["000001.SZ"],
            "close": [10.0]
        })

        result = mock_service._process_raw_df(df)
        assert "_DATE_" in result.columns
        assert "_ASSET_" in result.columns
        assert result["_DATE_"].dtype == object

    def test_process_raw_df_asset_mapping(self, mock_service):
        """资产 ID 映射"""
        # 首先添加一些资产到管理器
        snap = pl.DataFrame({
            "asset": ["000001.SZ"],
            "name": ["平安"],
            "list_date": [date(2020, 1, 1)],
            "delist_date": [None],
            "exchange": ["SSE"],
        }).cast(mock_service.assets_mgr.schema)

        mock_service.assets_mgr.update_assets(snap)

        # 现在测试映射
        df = pd.DataFrame({
            "trade_date": ["20230101"],
            "ts_code": ["000001.SZ"],
            "close": [10.0]
        })

        result = mock_service._process_raw_df(df)
        assert "asset_id" in result.columns
        assert result["asset_id"][0] == 0

    def test_process_raw_df_unmapped_asset(self, mock_service):
        """未知资产处理"""
        df = pd.DataFrame({
            "trade_date": ["20230101"],
            "ts_code": ["999999.SZ"],  # 不存在的资产
            "close": [10.0]
        })

        result = mock_service._process_raw_df(df)
        # 如果资产不在映射中且映射为空，asset_id 列不会被创建
        assert "_DATE_" in result.columns
        assert "_ASSET_" in result.columns


# ============================================================================
# 集成测试
# ============================================================================

class TestDataProviderIntegration:
    """数据提供者集成测试"""

    def test_cache_and_load_workflow(self):
        """缓存和读取工作流"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_mgr = HDF5CacheManager(Path(tmpdir) / "cache")

            # 保存数据
            df = pd.DataFrame({
                "trade_date": ["2023-01-01"],
                "ts_code": ["000001.SZ"],
                "close": [10.0],
                "volume": [1000.0]
            })

            cache_mgr.save_to_hdf5("daily", "20230101", df)

            # 检查缓存
            assert cache_mgr.is_cached("daily", "20230101")

            # 加载数据
            loaded = cache_mgr.load_from_hdf5("daily", ["20230101"])
            assert loaded is not None
            assert len(loaded) == 1

    def test_rate_limiter_integration(self):
        """限流器集成"""
        rl = RateLimiter(is_vip=True)
        import time

        times = []
        for _ in range(3):
            t = time.time()
            rl.wait()
            times.append(t)

        # 检查间隔时间
        interval1 = times[1] - times[0]
        interval2 = times[2] - times[1]

        # 首次 wait 没有等待（last_request_time = 0），后续有等待
        assert interval2 >= 0.07  # 至少 75ms

    def test_schema_validation_workflow(self):
        """Schema 验证工作流"""
        provider = DataProvider()

        # 有效的 Schema
        valid_lf = pl.LazyFrame({
            "_DATE_": [date(2023, 1, 1)],
            "_ASSET_": ["000001.SZ"],
            "CLOSE": [10.0],
            "VOLUME": [1000.0],
            "IS_SUSPENDED": [False],
        })
        assert provider.validate_schema(valid_lf) is True

        # 无效的 Schema（缺少列）
        invalid_lf = pl.LazyFrame({
            "_DATE_": [date(2023, 1, 1)],
            "_ASSET_": ["000001.SZ"],
        })
        assert provider.validate_schema(invalid_lf) is False


# ============================================================================
# 边界情况测试
# ============================================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_empty_dataframe_handling(self):
        """空 DataFrame 处理"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_mgr = HDF5CacheManager(Path(tmpdir))

            df = pd.DataFrame()  # 空 DataFrame
            cache_mgr.save_to_hdf5("daily", "20230101", df)

            # 应该不保存空数据
            result = cache_mgr.load_from_hdf5("daily", ["20230101"])
            assert result is None

    def test_null_date_handling(self):
        """空日期处理"""
        df = pd.DataFrame({
            "trade_date": [None, "2023-01-01"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "close": [10.0, 11.0]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(vars(settings), {
                'TUSHARE_TOKEN': 'mock_token',
                'RAW_DATA_DIR': Path(tmpdir) / 'raw',
                'WAREHOUSE_DIR': Path(tmpdir) / 'warehouse',
                'is_vip': True,
            }):
                with patch('alpha.data_provider.tushare_service.TushareDataService._init_tushare'):
                    service = TushareDataService()
                    result = service._process_raw_df(df)

                    # 空日期应该被转为 NaT 或 None
                    assert "_DATE_" in result.columns

    def test_invalid_date_format(self):
        """无效的日期格式"""
        provider = DataProvider()

        # 错误的日期格式应该抛出异常
        with pytest.raises(ValueError):
            provider.load_data("2023/01/01", "20230131")

    def test_large_dataframe_handling(self):
        """大 DataFrame 处理"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_mgr = HDF5CacheManager(Path(tmpdir))

            # 创建较大的 DataFrame（10000 行）
            df = pd.DataFrame({
                "trade_date": ["2023-01-01"] * 10000,
                "ts_code": [f"{i:06d}.SZ" for i in range(10000)],
                "close": [10.0 + i * 0.01 for i in range(10000)]
            })

            cache_mgr.save_to_hdf5("daily", "20230101", df)
            assert cache_mgr.is_cached("daily", "20230101")

            loaded = cache_mgr.load_from_hdf5("daily", ["20230101"])
            assert loaded is not None
            assert len(loaded) == 10000
