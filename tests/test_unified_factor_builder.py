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


def test_is_st_from_daily_names_rules() -> None:
    class _DummyAssetsMgr:
        stock_type = pl.String

        @staticmethod
        def get_all_codes() -> list[str]:
            return [
                "000001.SZ",
                "000002.SZ",
                "000003.SZ",
                "000004.SZ",
                "000005.SZ",
                "000006.SZ",
            ]

    class _DummyCacheMgr:
        @staticmethod
        def load_as_polars(source: str, trading_dates: list[date]) -> pl.DataFrame:
            assert source == "daily_names"
            assert len(trading_dates) == 1
            return pl.DataFrame(
                {
                    F.DATE: [trading_dates[0]] * 6,
                    F.ASSET: [
                        "000001.SZ",
                        "000002.SZ",
                        "000003.SZ",
                        "000004.SZ",
                        "000005.SZ",
                        "000006.SZ",
                    ],
                    "name": [
                        "*ST中珠",
                        "st华微",
                        " 平安银行 ",
                        "中航产融退",
                        "示例退市",
                        None,
                    ],
                }
            )

    builder = UnifiedFactorBuilder.__new__(UnifiedFactorBuilder)
    builder.assets_mgr = _DummyAssetsMgr()
    builder.cache_manager = _DummyCacheMgr()

    result = (
        builder._op_clean_st([date(2022, 1, 3)])
        .collect()
        .sort(F.ASSET)
        .select([F.ASSET, F.IS_ST])
    )

    assert result[F.IS_ST].to_list() == [True, True, False, True, True, False]


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


def test_is_st_from_daily_names_precomputed_column() -> None:
    class _DummyAssetsMgr:
        stock_type = pl.String

        @staticmethod
        def get_all_codes() -> list[str]:
            return ["000001.SZ", "000002.SZ", "000003.SZ"]

    class _DummyCacheMgr:
        @staticmethod
        def load_as_polars(source: str, trading_dates: list[date]) -> pl.DataFrame:
            assert source == "daily_names"
            return pl.DataFrame(
                {
                    F.DATE: [trading_dates[0]] * 3,
                    F.ASSET: ["000001.SZ", "000002.SZ", "000003.SZ"],
                    "is_st": [True, False, None],
                }
            )

    builder = UnifiedFactorBuilder.__new__(UnifiedFactorBuilder)
    builder.assets_mgr = _DummyAssetsMgr()
    builder.cache_manager = _DummyCacheMgr()

    result = (
        builder._op_clean_st([date(2022, 1, 3)])
        .collect()
        .sort(F.ASSET)
        .select([F.ASSET, F.IS_ST])
    )

    assert result[F.IS_ST].to_list() == [True, False, False]


def test_april_disclosure_signal_binary_rules() -> None:
    builder = UnifiedFactorBuilder.__new__(UnifiedFactorBuilder)

    lf = pl.DataFrame(
        {
            F.DATE: [
                date(2022, 4, 18),
                date(2022, 4, 19),
                date(2022, 4, 20),
                date(2022, 4, 21),
                date(2022, 5, 5),
                date(2023, 1, 3),
                date(2022, 4, 18),
                date(2022, 4, 29),
                date(2022, 5, 5),
            ],
            F.ASSET: [
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000001.SZ",
                "000002.SZ",
                "000002.SZ",
                "000002.SZ",
            ],
            "_TMP_SUSPEND_": [False] * 9,
            F.IS_ST: [False] * 9,
            F.OPEN_RAW: [10.0] * 9,
            F.HIGH_RAW: [10.5] * 9,
            F.LOW_RAW: [9.5] * 9,
            F.CLOSE_RAW: [10.0] * 9,
            F.VWAP_RAW: [10.0] * 9,
            F.ADJ_FACTOR: [1.0] * 9,
            F.VOLUME: [100.0] * 9,
            F.AMOUNT: [1000.0] * 9,
            F.TOTAL_MV: [1.0] * 9,
            F.CIRC_MV: [1.0] * 9,
            F.PE: [1.0] * 9,
            F.PB: [1.0] * 9,
            F.PS: [1.0] * 9,
            F.TURNOVER_RATE: [1.0] * 9,
            F.UP_LIMIT: [11.0] * 9,
            F.DOWN_LIMIT: [9.0] * 9,
            "_TMP_APRIL_ANN_DATE": [
                date(2022, 4, 20),
                date(2022, 4, 20),
                date(2022, 4, 20),
                date(2022, 4, 20),
                None,
                None,
                None,
                None,
                None,
            ],
        }
    ).lazy()

    result = builder._op_process_indicators(lf).collect().sort([F.ASSET, F.DATE])

    s1 = result.filter(pl.col(F.ASSET) == "000001.SZ").select(
        [F.DATE, F.APRIL_DISCLOSURE_SIGNAL]
    )
    # 💡 改为：1-3月为true（有风险）；披露日当天为false；大于披露日才为true
    assert s1[F.APRIL_DISCLOSURE_SIGNAL].to_list() == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]

    s2 = result.filter(pl.col(F.ASSET) == "000002.SZ").select(
        [F.DATE, F.APRIL_DISCLOSURE_SIGNAL]
    )
    assert s2[F.APRIL_DISCLOSURE_SIGNAL].to_list() == [False, False, False]


def test_clean_disclosure_uses_latest_ann_date_per_asset_year() -> None:
    class _DummyAssetsMgr:
        stock_type = pl.String

        @staticmethod
        def get_all_codes() -> list[str]:
            return ["000001.SZ"]

    class _DummyCacheMgr:
        @staticmethod
        def load_as_polars(source: str, trading_dates: list[date]) -> pl.DataFrame:
            assert source == "disclosure_date"
            assert trading_dates == [date(2021, 12, 31)]
            return pl.DataFrame(
                {
                    F.DATE: [date(2021, 12, 31)] * 3,
                    F.ASSET: ["000001.SZ", "000001.SZ", "000001.SZ"],
                    "ann_date": ["20220420", "20220428", "20220506"],
                    "actual_date": [
                        "20220430",
                        "20220505",
                        "20220510",
                    ],  # 优先使用 actual_date
                }
            )

    builder = UnifiedFactorBuilder.__new__(UnifiedFactorBuilder)
    builder.assets_mgr = _DummyAssetsMgr()
    builder.cache_manager = _DummyCacheMgr()

    result = (
        builder._op_clean_disclosure([date(2022, 4, 26), date(2022, 4, 29)])
        .collect()
        .sort(F.DATE)
    )

    # 💡 现在优先使用 actual_date，最大值是 20220510（2022-05-10）
    assert result["_TMP_APRIL_ANN_DATE"].to_list() == [
        date(2022, 5, 10),
        date(2022, 5, 10),
    ]
