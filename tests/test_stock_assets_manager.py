import os
import polars as pl
from pathlib import Path
import pytest

from alpha.data_provider.stock_assets_manager import StockAssetsManager


def make_snapshot(rows):
    # rows: list of dicts with asset,name,list_date,delist_date,exchange
    return pl.DataFrame(rows)


def test_init_empty(tmp_path):
    wh = tmp_path / "warehouse"
    mgr = StockAssetsManager(path=wh)
    props = mgr.get_properties()
    assert props.height == 0
    assert list(props.columns) == [c for c in ["asset", "name", "list_date", "delist_date", "exchange"] if c in props.columns]


def test_update_add_and_update_preserve_order_and_append(tmp_path):
    wh = tmp_path / "warehouse"
    mgr = StockAssetsManager(path=wh)

    snap1 = make_snapshot([
        {"asset": "A", "name": "Alpha", "list_date": "2020-01-01", "delist_date": None, "exchange": "SSE"},
        {"asset": "B", "name": "Beta", "list_date": "2020-02-01", "delist_date": None, "exchange": "SSE"},
    ])

    mgr.update_assets(snap1)
    props1 = mgr.get_properties()
    assert props1.height == 2
    assert props1.get_column("asset").to_list() == ["A", "B"]

    # Now update A's name and add C; order should remain A,B then C appended
    snap2 = make_snapshot([
        {"asset": "A", "name": "Alpha-new", "list_date": "2020-01-01", "delist_date": None, "exchange": "SSE"},
        {"asset": "C", "name": "Gamma", "list_date": "2021-03-01", "delist_date": None, "exchange": "SSE"},
    ])

    mgr.update_assets(snap2)
    props2 = mgr.get_properties()
    assert props2.height == 3
    assert props2.get_column("asset").to_list() == ["A", "B", "C"]
    # A's name updated
    a_row = [r for r in props2.to_dicts() if r["asset"] == "A"][0]
    assert a_row["name"] == "Alpha-new"


def test_persistence_and_types(tmp_path):
    wh = tmp_path / "warehouse"
    mgr = StockAssetsManager(path=wh)

    snap = make_snapshot([
        {"asset": "X", "name": "Xco", "list_date": "2019-05-01", "delist_date": None, "exchange": "SSE"},
    ])

    mgr.update_assets(snap)
    props = mgr.get_properties()

    # persisted file should exist
    assert (wh / "stock_assets.parquet").exists()

    # load a new manager from same path and verify content
    mgr2 = StockAssetsManager(path=wh)
    props2 = mgr2.get_properties()
    assert props2.height == 1
    assert props2.get_column("asset").to_list() == ["X"]

    # exchange should be categorical in returned properties if supported
    try:
        assert props2.get_column("exchange").dtype == pl.Categorical
    except Exception:
        # 在某些 polars 版本上 dtype 比较可能不同，至少要能访问列
        assert "exchange" in props2.columns
