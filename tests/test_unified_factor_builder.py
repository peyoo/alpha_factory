from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from alpha_factory.data_provider.unified_factor_builder import UnifiedFactorBuilder
from alpha_factory.utils.schema import F


def test_is_st_keeps_true_after_first_true() -> None:
    builder = UnifiedFactorBuilder.__new__(UnifiedFactorBuilder)

    lf = pl.DataFrame(
        {
            F.DATE: [
                date(2022, 1, 3),
                date(2022, 1, 4),
                date(2022, 1, 5),
                date(2022, 1, 6),
            ],
            F.ASSET: ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "_TMP_SUSPEND_": [False, False, False, False],
            F.IS_ST: [False, True, None, False],
            F.OPEN_RAW: [10.0, 10.0, 10.0, 10.0],
            F.HIGH_RAW: [10.5, 10.5, 10.5, 10.5],
            F.LOW_RAW: [9.5, 9.5, 9.5, 9.5],
            F.CLOSE_RAW: [10.0, 10.0, 10.0, 10.0],
            F.VWAP_RAW: [10.0, 10.0, 10.0, 10.0],
            F.ADJ_FACTOR: [1.0, 1.0, 1.0, 1.0],
            F.VOLUME: [100.0, 100.0, 100.0, 100.0],
            F.AMOUNT: [1000.0, 1000.0, 1000.0, 1000.0],
            F.TOTAL_MV: [1.0, 1.0, 1.0, 1.0],
            F.CIRC_MV: [1.0, 1.0, 1.0, 1.0],
            F.PE: [1.0, 1.0, 1.0, 1.0],
            F.PB: [1.0, 1.0, 1.0, 1.0],
            F.PS: [1.0, 1.0, 1.0, 1.0],
            F.TURNOVER_RATE: [1.0, 1.0, 1.0, 1.0],
            F.UP_LIMIT: [11.0, 11.0, 11.0, 11.0],
            F.DOWN_LIMIT: [9.0, 9.0, 9.0, 9.0],
        }
    ).lazy()

    result = builder._op_process_indicators(lf).collect().sort(F.DATE)

    assert result[F.IS_ST].to_list() == [False, True, True, True]


def test_datas_end_none_queries_to_latest(tmp_path: Path) -> None:
    factor_dir = tmp_path / "unified_factors"
    factor_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            F.DATE: [date(2024, 1, 2), date(2024, 1, 3)],
            F.ASSET: ["000001.SZ", "000002.SZ"],
            F.CLOSE: [10.0, 20.0],
        }
    ).write_parquet(factor_dir / "2024.parquet")

    pl.DataFrame(
        {
            F.DATE: [date(2025, 1, 2), date(2025, 1, 3)],
            F.ASSET: ["000001.SZ", "000001.SZ"],
            F.CLOSE: [11.0, 12.0],
        }
    ).write_parquet(factor_dir / "2025.parquet")

    builder = UnifiedFactorBuilder.__new__(UnifiedFactorBuilder)
    builder.warehouse_dir = tmp_path

    result = builder.datas(
        start=date(2024, 1, 2),
        end=None,
        assets=["000001.SZ"],
        cols=[F.CLOSE],
    ).sort([F.DATE, F.ASSET])

    assert result.columns == [F.DATE, F.ASSET, F.CLOSE]
    assert result[F.DATE].to_list() == [
        date(2024, 1, 2),
        date(2025, 1, 2),
        date(2025, 1, 3),
    ]
    assert result[F.CLOSE].to_list() == [10.0, 11.0, 12.0]
