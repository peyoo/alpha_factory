from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.utils.schema import F


class DummyTushareService:
    def get_latest_date_from_warehouse(self) -> str:
        return "20240131"


class DummyPoolA:
    def __init__(self) -> None:
        self.label_col_funcs = []

    def pool(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    def extra_cols(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    def needed_cols(self) -> list[str]:
        return ["POOL_MASK"]

    def preprocessor(
        self,
        df: pl.LazyFrame,
        factors: list[str],
    ) -> pl.LazyFrame:
        return df


class DummyPoolB:
    def __init__(self) -> None:
        self.label_col_funcs = []

    def pool(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    def extra_cols(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    def needed_cols(self) -> list[str]:
        return ["POOL_MASK"]

    def preprocessor(
        self,
        df: pl.LazyFrame,
        factors: list[str],
    ) -> pl.LazyFrame:
        return df


def _make_provider() -> DataProvider:
    provider = DataProvider.__new__(DataProvider)
    provider.tushare_service = DummyTushareService()
    return provider


def _make_base_lf() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            F.DATE: [date(2024, 1, 2)],
            F.ASSET: ["000001.SZ"],
            "POOL_MASK": [True],
        }
    ).lazy()


def test_pool_cache_key_changes_when_pool_source_changes() -> None:
    provider = _make_provider()
    start, end = date(2024, 1, 1), date(2024, 1, 31)

    path_a = provider._build_pool_cache_path(DummyPoolA(), start, end)
    path_b = provider._build_pool_cache_path(DummyPoolB(), start, end)

    assert path_a != path_b


def test_full_cache_key_is_order_independent() -> None:
    provider = _make_provider()
    fake_pool_path = Path("/tmp/pool_base_abc.parquet")

    path_1 = provider._build_factors_cache_path(
        fake_pool_path, ["A = CLOSE", "B = OPEN", "A = CLOSE"], "md5"
    )
    path_2 = provider._build_factors_cache_path(
        fake_pool_path, ["B = OPEN", "  A = CLOSE  "], "md5"
    )

    assert path_1 == path_2


def test_load_pool_data_md5_uses_two_stage_and_no_final_cache() -> None:
    provider = _make_provider()
    pool = DummyPoolA()
    base_calls: list = []
    factors_calls: list = []
    fake_pool_path = Path("/tmp") / "pool_base_abc123.parquet"

    def fake_build_pool_base_data(p, start, end):
        base_calls.append((p, start, end))
        return _make_base_lf(), fake_pool_path

    def fake_build_factors_view(p, lf, exprs, processors=None, final_cache_path=None):
        factors_calls.append({"final_cache_path": final_cache_path})
        return lf.with_columns(pl.lit(1.0).alias("FACTOR_X"))

    provider._build_pool_base_data = fake_build_pool_base_data  # type: ignore[method-assign]
    provider._build_factors_view = fake_build_factors_view  # type: ignore[method-assign]

    lf = provider.load_pool_data(
        pool=pool,
        start_date="20240101",
        end_date="20240131",
        exprs=["FACTOR_X = CLOSE"],
        cache="md5",
    )

    assert len(base_calls) == 1
    # cache="md5" → factors 缓存路径应自动合成且以 factor_data_ 开头
    factors_cache = factors_calls[0]["final_cache_path"]
    assert factors_cache is not None
    assert factors_cache.name.startswith("factor_data_")

    columns = set(lf.collect_schema().names())
    assert {F.DATE, F.ASSET, "POOL_MASK", "FACTOR_X"}.issubset(columns)


def test_load_pool_data_explicit_cache_with_exprs_uses_final_key(
    tmp_path: Path,
) -> None:
    provider = _make_provider()
    pool = DummyPoolA()
    captured: dict[str, Path] = {}
    fake_pool_path = Path("/tmp") / "pool_base_abc123.parquet"

    def fake_load_cached_lazyframe(cache_path):
        return None

    def fake_build_pool_base_data(p, start, end):
        return _make_base_lf(), fake_pool_path

    def fake_apply_column_exprs(lf, exprs, codegen_over_null=None):
        return lf.with_columns(pl.lit(1.0).alias("A")), ["A"]

    def fake_persist_cache_and_reload(lf, cache_path):
        captured["cache_path"] = cache_path
        return lf

    provider._load_cached_lazyframe = fake_load_cached_lazyframe  # type: ignore[method-assign]
    provider._build_pool_base_data = fake_build_pool_base_data  # type: ignore[method-assign]
    provider._apply_column_exprs = fake_apply_column_exprs  # type: ignore[method-assign]
    provider._persist_cache_and_reload = fake_persist_cache_and_reload  # type: ignore[method-assign]

    exprs = ["B = OPEN", "A = CLOSE"]
    provider.load_pool_data(
        pool=pool,
        start_date="20240101",
        end_date="20240131",
        exprs=exprs,
        cache=tmp_path,
    )

    # cache=显式路径 → 直接 resolve() 使用，不附加 hash 子目录
    assert captured["cache_path"] == tmp_path.resolve()
