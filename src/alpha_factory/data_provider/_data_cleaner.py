"""_data_cleaner.py — UnifiedFactorBuilder 的清洗算子 Mixin。

DataCleanerMixin 包含 8 个 _op_clean_* 方法和 _op_process_indicators，
依赖宿主类提供 cache_manager, assets_mgr, _valid_asset_codes 属性。
"""

from __future__ import annotations

from datetime import date
from typing import List

import polars as pl
import polars.selectors as cs

from alpha_factory.utils.schema import F


class DataCleanerMixin:
    """Mixin：将行情/基本面原始数据清洗为统一 Lazy 列。

    宿主类必须提供:
    - self.cache_manager: HDF5CacheManager
    - self.assets_mgr: StockAssetsManager
    - self._valid_asset_codes: list[str]
    """

    def _ensure_valid_assets(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """防火墙：仅剔除名录外代码。"""
        if not hasattr(self, "_valid_asset_codes"):
            self._valid_asset_codes = self.assets_mgr.get_all_codes()  # type: ignore[attr-defined]
        return lf.filter(pl.col(F.ASSET).is_in(self._valid_asset_codes))  # type: ignore[attr-defined]

    def _op_clean_daily(self, trading_dates: List[date]) -> pl.LazyFrame:
        """清洗原始行情：使用 load_as_polars 获取数据"""
        df_pl = self.cache_manager.load_as_polars("daily", trading_dates)  # type: ignore[attr-defined]
        if df_pl is None:
            return pl.LazyFrame(schema={F.DATE: pl.Date, F.ASSET: pl.String})

        return self._ensure_valid_assets(df_pl.lazy()).select(
            [
                pl.col(F.DATE),
                pl.col(F.ASSET),
                pl.col("open").cast(pl.Float32).alias(F.OPEN_RAW),
                pl.col("high").cast(pl.Float32).alias(F.HIGH_RAW),
                pl.col("low").cast(pl.Float32).alias(F.LOW_RAW),
                pl.col("close").cast(pl.Float32).alias(F.CLOSE_RAW),
                (pl.col("vol") * 100).cast(pl.Float32).alias(F.VOLUME),
                (pl.col("amount") * 1000).cast(pl.Float32).alias(F.AMOUNT),
                pl.when(pl.col("vol") > 0)
                .then(
                    (pl.col("amount") * 1000 / (pl.col("vol") * 100)).cast(pl.Float32)
                )
                .otherwise(None)
                .alias(F.VWAP_RAW),
            ]
        )

    def _op_clean_adj(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("adj_factor", trading_dates)  # type: ignore[attr-defined]
        if df_pl is None:
            return pl.LazyFrame()

        return self._ensure_valid_assets(df_pl.lazy()).select(
            [
                pl.col(F.DATE),
                pl.col(F.ASSET),
                pl.col("adj_factor").cast(pl.Float32).alias("ADJ_FACTOR"),
            ]
        )

    def _op_clean_basic(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("daily_basic", trading_dates)  # type: ignore[attr-defined]
        if df_pl is None:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: pl.String,
                    "PE": pl.Float32,
                    "PB": pl.Float32,
                    "PS": pl.Float32,
                    "TURNOVER_RATE": pl.Float32,
                    "TOTAL_MV": pl.Float64,
                    "CIRC_MV": pl.Float64,
                }
            )

        return self._ensure_valid_assets(df_pl.lazy()).select(
            [
                pl.col(F.DATE),
                pl.col(F.ASSET),
                pl.col("pe").cast(pl.Float32).alias(F.PE),
                pl.col("pb").cast(pl.Float32).alias(F.PB),
                pl.col("ps").cast(pl.Float32).alias(F.PS),
                pl.col("turnover_rate").cast(pl.Float32).alias(F.TURNOVER_RATE),
                (pl.col("total_mv") * 10000).cast(pl.Float64).alias(F.TOTAL_MV),
                (pl.col("circ_mv") * 10000).cast(pl.Float64).alias(F.CIRC_MV),
            ]
        )

    def _op_clean_limit(self, trading_dates: List[date]) -> pl.LazyFrame:
        df_pl = self.cache_manager.load_as_polars("stk_limit", trading_dates)  # type: ignore[attr-defined]
        if df_pl is None:
            return pl.LazyFrame()
        return self._ensure_valid_assets(df_pl.lazy()).select(
            [
                pl.col(F.DATE),
                pl.col(F.ASSET),
                pl.col("up_limit").cast(pl.Float32).alias(F.UP_LIMIT),
                pl.col("down_limit").cast(pl.Float32).alias(F.DOWN_LIMIT),
            ]
        )

    def _op_clean_suspend(self, trading_dates: List[date]) -> pl.LazyFrame:
        """清洗显式停牌数据"""
        df_pl = self.cache_manager.load_as_polars("suspend_d", trading_dates)  # type: ignore[attr-defined]
        if df_pl is None:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: pl.String,
                    "_TMP_SUSPEND_": pl.Boolean,
                }
            )

        return self._ensure_valid_assets(df_pl.lazy()).select(
            [pl.col(F.DATE), pl.col(F.ASSET), pl.lit(True).alias("_TMP_SUSPEND_")]
        )

    def _op_clean_st(self, trading_dates: List[date]) -> pl.LazyFrame:
        """加载融合的 ST 标记因子，使用 forward_fill 填充缺失值。"""
        df_st = self.cache_manager.load_as_polars("st", trading_dates)  # type: ignore[attr-defined]
        if df_st is None or "is_st" not in df_st.columns:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: pl.String,
                    F.IS_ST: pl.Boolean,
                }
            )

        lf = self._ensure_valid_assets(df_st.lazy())
        return (
            lf.sort([F.ASSET, F.DATE])
            .with_columns(
                pl.col("is_st")
                .forward_fill()
                .over(F.ASSET)
                .fill_null(False)
                .cast(pl.Boolean)
                .alias(F.IS_ST)
            )
            .select([pl.col(F.DATE), pl.col(F.ASSET), pl.col(F.IS_ST)])
        )

    def _op_clean_disclosure(self, trading_dates: List[date]) -> pl.LazyFrame:
        """加载预计算的披露因子：提取 flag 列，None 填充为 False。"""
        if not trading_dates:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: pl.String,
                    "flag": pl.Boolean,
                }
            )

        df_pl = self.cache_manager.load_as_polars("disclosure", trading_dates)  # type: ignore[attr-defined]
        if df_pl is None:
            return pl.LazyFrame(
                schema={
                    F.DATE: pl.Date,
                    F.ASSET: pl.String,
                    "flag": pl.Boolean,
                }
            )

        return self._ensure_valid_assets(df_pl.lazy()).with_columns(
            pl.col("flag").fill_null(False)
        )

    def _op_process_indicators(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """核心业务逻辑：填充、状态判定、复权计算"""
        schema_names = set(lf.collect_schema().names())
        additional_cols: list[pl.Expr] = []
        if "flag" not in schema_names:
            additional_cols.append(pl.lit(None).cast(pl.Boolean).alias("flag"))

        ffill_cols = [
            F.CLOSE_RAW,
            F.ADJ_FACTOR,
            F.TOTAL_MV,
            F.CIRC_MV,
            F.PE,
            F.PB,
            F.PS,
            F.TURNOVER_RATE,
            F.VWAP_RAW,
            F.UP_LIMIT,
            F.DOWN_LIMIT,
        ]

        result = (
            lf.sort([F.ASSET, F.DATE])
            .with_columns(additional_cols)
            .with_columns(
                [
                    (
                        pl.col("_TMP_SUSPEND_").fill_null(False)
                        | pl.col(F.CLOSE_RAW).is_null()
                    ).alias(F.IS_SUSPENDED),
                    pl.col(F.IS_ST)
                    .fill_null(False)
                    .cast(pl.Int8)
                    .cum_max()
                    .over(F.ASSET)
                    .cast(pl.Boolean)
                    .alias(F.IS_ST),
                    pl.col("flag").cast(pl.Boolean).alias(F.APRIL_DISCLOSURE_SIGNAL),
                    pl.col([F.VOLUME, F.AMOUNT]).fill_null(0.0),
                ]
            )
            .with_columns(
                [
                    pl.col(ffill_cols).forward_fill().over(F.ASSET),
                    pl.col(F.APRIL_DISCLOSURE_SIGNAL).forward_fill().over(F.ASSET),
                ]
            )
            .with_columns(
                [
                    pl.col(F.OPEN_RAW).fill_null(pl.col(F.CLOSE_RAW)),
                    pl.col(F.HIGH_RAW).fill_null(pl.col(F.CLOSE_RAW)),
                    pl.col(F.LOW_RAW).fill_null(pl.col(F.CLOSE_RAW)),
                    pl.col(F.VWAP_RAW).fill_null(pl.col(F.CLOSE_RAW)),
                ]
            )
            .with_columns(
                [
                    (pl.col(F.OPEN_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.OPEN),
                    (pl.col(F.HIGH_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.HIGH),
                    (pl.col(F.LOW_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.LOW),
                    (pl.col(F.CLOSE_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.CLOSE),
                    (pl.col(F.VWAP_RAW) * pl.col(F.ADJ_FACTOR))
                    .cast(pl.Float32)
                    .alias(F.VWAP),
                ]
            )
        )

        # 清理临时列
        return result.drop([cs.starts_with("_TMP_")])
