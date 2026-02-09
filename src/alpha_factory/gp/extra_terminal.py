import polars as pl

from alpha_factory.utils.schema import F


def add_extra_terminals(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        # 必须排序，确保 shift(1) 拿到的是前一交易日
        lf.sort([F.ASSET, F.DATE])
        .with_columns([
            pl.col(F.TOTAL_MV).log().alias("LOG_MV"),
            # 先算 RET
            (pl.col(F.CLOSE) / pl.col(F.CLOSE).shift(1) - 1).over(F.ASSET).alias("RET"),
            (pl.col("VWAP") / pl.col("VWAP").shift(1) - 1).over(F.ASSET).alias("VWAP_RET"),
        ])
        .with_columns([
            # 直接复用上面算好的 RET，减少计算量
            (pl.col("RET").abs() / pl.col("AMOUNT") * 1e6).alias("ILLIQ")
        ])
        # 建议只在 collect 之后或必要时填充，或者使用这种方式：
        .fill_nan(None)
    )
