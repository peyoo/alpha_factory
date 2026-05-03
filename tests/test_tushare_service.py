from __future__ import annotations

from datetime import date

import pandas as pd

from alpha_factory.data_provider.tushare_service import TushareDataService


class _DummyRateLimiter:
    @staticmethod
    def wait() -> None:
        return


class _DummyCacheManager:
    def __init__(self) -> None:
        self.calls: list[tuple] = []  # (source, df)

    @staticmethod
    def is_cached(source: str, trade_date: date) -> bool:
        # 只跳过已处理的数据源
        return source in (
            "daily",
            "adj_factor",
            "daily_basic",
            "stk_limit",
            "suspend_d",
        )

    def save_to_hdf5(self, source: str, trade_date: date, df: pd.DataFrame) -> None:
        self.calls.append((source, df.copy()))

    @property
    def saved_source(self) -> str | None:
        """向后兼容：返回最后一次保存的数据源"""
        return self.calls[-1][0] if self.calls else None

    @property
    def saved_df(self) -> pd.DataFrame | None:
        """向后兼容：返回最后一次保存的 DataFrame"""
        return self.calls[-1][1] if self.calls else None


class _DummyCalendar:
    """Mock 交易日历，用于计算前一天"""

    def offset(self, trade_date: date, delta: int) -> date:
        # 简单实现：假设每天都是交易日
        from datetime import timedelta

        return trade_date + timedelta(days=delta)


class _DummyPro:
    @staticmethod
    def daily(*args, **kwargs):
        raise AssertionError("daily should not be called when cached")

    @staticmethod
    def adj_factor(*args, **kwargs):
        raise AssertionError("adj_factor should not be called when cached")

    @staticmethod
    def daily_basic(*args, **kwargs):
        raise AssertionError("daily_basic should not be called when cached")

    @staticmethod
    def stk_limit(*args, **kwargs):
        raise AssertionError("stk_limit should not be called when cached")

    @staticmethod
    def suspend_d(*args, **kwargs):
        raise AssertionError("suspend_d should not be called when cached")

    @staticmethod
    def namechange(
        *, start_date: str, end_date: str, fields: list[str]
    ) -> pd.DataFrame:
        """Mock namechange API：返回名称变更记录"""
        assert fields == ["ts_code", "name", "change_reason"]
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["平安银行"],
                "change_reason": ["其他"],
            }
        )

    @staticmethod
    def stock_st(*, trade_date: str, fields: list[str]) -> pd.DataFrame:
        """Mock stock_st API - 返回 ST 股票列表"""
        assert fields == ["ts_code"]
        # 返回空结果（以测试 namechange 规则的优先级）
        return pd.DataFrame({"ts_code": []})

    @staticmethod
    def disclosure_date(*, end_date: str, fields: list[str]) -> pd.DataFrame:
        assert end_date == "20211231"
        assert fields == [
            "ts_code",
            "ann_date",
            "end_date",
            "pre_date",
            "actual_date",
            "modify_date",
        ]
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "ann_date": ["20220420"],
                "end_date": ["20211231"],
                "pre_date": ["20220420"],
                "actual_date": ["20220420"],
                "modify_date": [""],
            }
        )


def test_sync_single_day_bundle_st_via_st_data() -> None:
    """验证 _sync_single_day_bundle 中 st 数据由 _st_data 方法生成"""
    service = TushareDataService.__new__(TushareDataService)
    service.rate_limiter = _DummyRateLimiter()
    service.pro = _DummyPro()
    service.cache_manager = _DummyCacheManager()
    service.calendar = _DummyCalendar()

    service._sync_single_day_bundle(date(2022, 1, 3), idx=1, total=1)

    # 验证保存了 st
    saved_sources = [call[0] for call in service.cache_manager.calls]
    assert "st" in saved_sources, f"Expected 'st' in {saved_sources}"

    # 验证 st 数据包含 is_st 列
    st_df = next(df for source, df in service.cache_manager.calls if source == "st")
    assert "is_st" in st_df.columns
    assert "ts_code" in st_df.columns
    assert len(st_df) > 0
