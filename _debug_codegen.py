"""Debug script: verify codegen_exec actually adds columns to LazyFrame schema."""

import polars as pl
from datetime import date

from alpha_factory.data_provider.data_provider import DataProvider
from alpha_factory.data_provider.pool import MainSmallPool
from expr_codegen.tool import codegen_exec
from alpha_factory.config.base import settings

dp = DataProvider()
pool = MainSmallPool()
pool_data, path = dp._build_pool_base_data(pool, date(2019, 1, 1), date(2026, 3, 6))
print("pool_data cols:", len(pool_data.collect_schema().names()))

result = codegen_exec(
    pool_data,
    "_RAW_f1 = ts_mean(AMOUNT,60)",
    "_RAW_f2 = TOTAL_MV",
    over_null=None,
    template_file=settings.template_path_str,
    date="DATE",
    asset="ASSET",
)
print("codegen_exec result type:", type(result))
if isinstance(result, pl.LazyFrame):
    cols = result.collect_schema().names()
    print("schema cols count:", len(cols))
    print("_RAW_f1 in schema:", "_RAW_f1" in cols)
    print("all schema cols:", cols)
    # Try to actually collect a small sample
    sample = result.head(2).collect()
    print("collected cols:", sample.columns)
    print("_RAW_f1 in collected:", "_RAW_f1" in sample.columns)
elif isinstance(result, tuple):
    print("tuple len:", len(result))
    for i, r in enumerate(result):
        print(f"  [{i}] type:{type(r)}")
        if isinstance(r, pl.LazyFrame):
            print("   lf cols:", r.collect_schema().names())
else:
    print("unexpected type:", type(result), result)
