import polars as pl
from polars_ta.wq import cs_mad_zscore_resid


def my_cs_mad_zscore_resid(x):
    return cs_mad_zscore_resid(x, pl.col("LOG_MV"))
