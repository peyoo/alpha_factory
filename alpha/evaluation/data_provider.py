"""
为因子的评估提供数据
"""
from expr_codegen import codegen_exec

from alpha.data_provider import DataProvider
import polars as pl


def create_tradable_micro_pool(
        df: pl.DataFrame,
        mkt_cap_col: str = "TOTAL_MV",
        amount_col: str = "AMOUNT",
        date_col: str = "DATE",
        asset_col: str = "ASSET",
        top_n_percent: float = 0.2,
        min_amount: float = 1e7,
        min_list_days: int = 242
) -> pl.DataFrame:
    """
    【完全实盘化】小微盘掩码：
    1. 动态市值百分比 + 流动性过滤
    2. 严格剔除 ST、停牌、新股
    3. 精准识别：封死涨停(不可买入) & 封死跌停(不可卖出)
    """

    return (
        df.with_columns([
            # 截面市值排名
            pl.col(mkt_cap_col).rank().over(date_col).alias("mv_rank"),
            pl.col(asset_col).count().over(date_col).alias("total_count"),

            # 精准判断封板逻辑：收盘价 == 涨/跌停价 且 成交额不为0（排除全天停牌）
            (pl.col("CLOSE_RAW") >= pl.col("UP_LIMIT")).alias("is_locked_up"),
            (pl.col("CLOSE_RAW") <= pl.col("DOWN_LIMIT")).alias("is_locked_down")
        ])
        .with_columns([
            (
                    (pl.col("mv_rank") / pl.col("total_count") <= top_n_percent) &
                    (pl.col("IS_ST") is False) &
                    (pl.col("IS_SUSPENDED") is False) &
                    (pl.col("LIST_DAYS") >= min_list_days) &
                    (pl.col(amount_col) >= min_amount) &
                    # 过滤掉无法买入的封死涨停股 和 无法卖出的封死跌停股
                    (pl.col("is_locked_up") is False) &
                    (pl.col("is_locked_down") is False)
            ).alias("POOL_TRADABLE")
        ])
    )


def prepare_data_for_single_factor(start, end, expr):
    """
    准备因子评估所需的数据：包含收益率计算、因子生成及数据清洗
    """
    data_provider = DataProvider()
    lf = data_provider.load_data(start, end)

    # --- 1. 计算 T+2 实盘收益率 ---
    # 逻辑：T日发出信号 -> T+1开盘买入 -> T+2开盘卖出
    lf = lf.sort(["ASSET", "DATE"])
    lf = lf.with_columns(
        (pl.col("OPEN").shift(-2).over("ASSET") /
         pl.col("OPEN").shift(-1).over("ASSET") - 1).alias("ret_real_trade")
    )

    # --- 2. 生成因子值 ---
    # 假设 codegen_exec 会处理 expr 并生成名为 factor_1 的列
    code = f"factor_1 = {expr}"
    df = codegen_exec(lf, code, over_null=None,date='DATE',asset='ASSET')

    # --- 3. 数据清洗 (关键步骤) ---
    # a. 必须滤除 ret_real_trade 为 null 的行（即每个 ASSET 的最后两天）
    # b. 必须滤除 factor_1 为 null 的行（因子计算窗口不足的初期数据）
    df = df.filter(
        pl.col("ret_real_trade").is_not_null() &
        pl.col("factor_1").is_not_null()
    )

    df = df.collect()

    return df


def prepare_data_for_batch_factors(start, end, expr_list = [],csv_file=None,expr_col="expression"):
    """
    准备批量因子评估所需的数据
    """
    data_provider = DataProvider()
    lf = data_provider.load_data(start, end)

    # --- 1. 计算 T+2 实盘收益率 ---
    lf = lf.sort(["ASSET", "DATE"])
    lf = lf.with_columns(
        (pl.col("OPEN").shift(-2).over("ASSET") /
         pl.col("OPEN").shift(-1).over("ASSET") - 1).alias("ret_real_trade")
    )

    # --- 2. 生成多个因子值 ---
    for idx, expr in enumerate(expr_list):
        code = f"factor_{idx+1} = {expr}"
        lf = codegen_exec(lf, code, over_null=None)

    # --- 3. 数据清洗 ---
    filter_cond = pl.col("ret_real_trade").is_not_null()
    for idx in range(len(expr_list)):
        filter_cond &= pl.col(f"factor_{idx+1}").is_not_null()

    df = lf.filter(filter_cond)

    return df
