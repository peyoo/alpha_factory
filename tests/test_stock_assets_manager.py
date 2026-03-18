"""
回归测试：StockAssetsManager.update_assets 在 ASSET 列被内部维护为
Categorical 之后，再次调用 update_assets 不应抛出 join key 类型不匹配异常。
"""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest

from alpha_factory.data_provider.stock_assets_manager import StockAssetsManager


def _make_snapshot(codes: list[str]) -> pl.DataFrame:
    """构造符合 ASSETS_SCHEMA 的最小快照 DataFrame（ASSET 为 Utf8）。"""
    return pl.DataFrame(
        {
            "ASSET": codes,
            "name": [f"Stock_{c}" for c in codes],
            "list_date": [datetime.date(2020, 1, 1)] * len(codes),
            "delist_date": [None] * len(codes),
            "exchange": ["SZ"] * len(codes),
            "market": ["主板"] * len(codes),
        },
        schema={
            "ASSET": pl.Utf8,
            "name": pl.Utf8,
            "list_date": pl.Date,
            "delist_date": pl.Date,
            "exchange": pl.Utf8,
            "market": pl.Utf8,
        },
    )


@pytest.fixture()
def tmp_manager(tmp_path: Path) -> StockAssetsManager:
    """返回一个使用临时目录、初始为空的 StockAssetsManager。"""
    return StockAssetsManager(path=tmp_path / "stock_assets.parquet")


class TestUpdateAssetsJoinKeyType:
    """验证 update_assets 的 join key 类型兼容性。"""

    def test_second_update_does_not_raise(self, tmp_manager: StockAssetsManager):
        """
        回归：第一次 update_assets 后 ASSET 被内部 cast 为 Categorical，
        第二次调用（新快照含 Utf8 类型 ASSET）不应抛出类型不匹配异常。
        """
        first_snapshot = _make_snapshot(["000001.SZ", "000002.SZ"])
        tmp_manager.update_assets(first_snapshot)

        # 内部 ASSET 现在是 Categorical
        assert tmp_manager._df.schema["ASSET"] == pl.Categorical

        # 第二次 update 包含新股票 + 旧股票，ASSET 仍为 Utf8 → 不应抛出
        second_snapshot = _make_snapshot(["000001.SZ", "000002.SZ", "000003.SZ"])
        tmp_manager.update_assets(second_snapshot)  # 不应抛出 join key 类型不匹配

        codes = set(tmp_manager._df.get_column("ASSET").cast(pl.Utf8).to_list())
        assert codes == {"000001.SZ", "000002.SZ", "000003.SZ"}

    def test_existing_assets_order_preserved(self, tmp_manager: StockAssetsManager):
        """已有资产行序不应因二次更新而改变（新资产追加到末尾）。"""
        first_snapshot = _make_snapshot(["000001.SZ", "000002.SZ"])
        tmp_manager.update_assets(first_snapshot)

        # 记录第一次 update 后的实际顺序（不假设 unique() 的输出顺序）
        after_first = tmp_manager._df.get_column("ASSET").cast(pl.Utf8).to_list()
        assert set(after_first) == {"000001.SZ", "000002.SZ"}

        second_snapshot = _make_snapshot(["000003.SZ", "000001.SZ", "000002.SZ"])
        tmp_manager.update_assets(second_snapshot)

        after_second = tmp_manager._df.get_column("ASSET").cast(pl.Utf8).to_list()
        # 旧资产相对顺序不变，新资产追加到末尾
        assert after_second[:2] == after_first
        assert after_second[2] == "000003.SZ"
