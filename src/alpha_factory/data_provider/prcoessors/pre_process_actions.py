import polars as pl
from polars_ta.wq import (
    cs_mad_zscore_resid,
    cs_mad_zscore_resid_zscore,
    cs_resid,
)


def cs_resid_log_mv(x):
    return cs_resid(x, pl.col("LOG_MV"))


def my_cs_mad_zscore_resid(x):
    return cs_mad_zscore_resid(x, pl.col("LOG_MV"))


def my_cs_mad_zscore_resid_zscore(x):
    return cs_mad_zscore_resid_zscore(x, pl.col("LOG_MV"))
