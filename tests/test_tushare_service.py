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
        self.saved_source: str | None = None
        self.saved_df: pd.DataFrame | None = None

    @staticmethod
    def is_cached(source: str, trade_date: date) -> bool:
        return source != "daily_names"

    def save_to_hdf5(self, source: str, trade_date: date, df: pd.DataFrame) -> None:
        self.saved_source = source
        self.saved_df = df.copy()


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
    def stock_st(*args, **kwargs):
        raise AssertionError("stock_st should not be called when cached")

    @staticmethod
    def bak_daily(*, trade_date: str, fields: list[str]) -> pd.DataFrame:
        assert trade_date == "20220103"
        assert fields == ["ts_code", "name"]
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["平安银行"],
            }
        )


def test_sync_single_day_bundle_daily_names_uses_bak_daily() -> None:
    service = TushareDataService.__new__(TushareDataService)
    service.rate_limiter = _DummyRateLimiter()
    service.pro = _DummyPro()
    service.cache_manager = _DummyCacheManager()

    service._sync_single_day_bundle(date(2022, 1, 3), idx=1, total=1)

    assert service.cache_manager.saved_source == "daily_names"
    assert service.cache_manager.saved_df is not None
    assert "name" not in service.cache_manager.saved_df.columns
    assert "is_st" in service.cache_manager.saved_df.columns
    assert not service.cache_manager.saved_df.loc[0, "is_st"]
