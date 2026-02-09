import polars as pl
from loguru import logger
from typing import Dict

# 添加项目路径
from alpha.data_provider import DataProvider
from alpha.data_provider.pool import main_small_pool
from alpha.evaluation.batch.full_metrics import batch_full_metrics
from alpha.gp.extra_terminal import add_extra_terminals
from alpha.gp.label import label_OO_for_tradable, label_OO_for_IC
from alpha.utils.schema import F


def run_batch_factor_test():
    # --- 1. 定义因子表达式映射 ---
    exprs = [
        "ts_mean(TURNOVER_RATE, 10)",
        "ts_mean(TURNOVER_RATE, 40)",
        "ts_mean(TURNOVER_RATE, 60)",
        "ts_mean(TURNOVER_RATE, 120)",
        "ts_std_dev(cs_mad_zscore_mask(VWAP), 30)",
    ]

    # 创建映射：factor_1 -> expression
    factor_expr_map: Dict[str, str] = {
        f"factor_{i}": expr for i, expr in enumerate(exprs)
    }
    exprs_with_names = [f"factor_{i}={expr}" for i, expr in enumerate(exprs)]

    # --- 2. 加载数据 ---
    logger.info("📡 正在从 DataProvider 加载数据并注入表达式...")
    lf = DataProvider().load_data(
        start_date="20190101",
        end_date="20251231",
        funcs=[main_small_pool, add_extra_terminals, label_OO_for_IC, label_OO_for_tradable],
        column_exprs=[f'{F.LABEL_FOR_RET}=OPEN[-2] / OPEN[-1] - 1', *exprs_with_names],
        lookback_window=200
    )

    # --- 3. 调用全维度评估 ---
    # fee 设为 0.003 以确保误差被保守覆盖
    logger.info("⚙️ 启动全维度指标评估 (Fee: 0.003)...")

    df_result = batch_full_metrics(
        lf,
        factors=r"^factor_.*",
        fee=0.0025,
        annual_days=252,
        n_bins= 10
    )

    # --- 4. 字段映射与后处理 ---
    # 匹配你提供的返回格式: factor, ic_ir, ann_ret, sharpe, turnover_est, direction
    df_result = (
        df_result
        .with_columns(
            pl.col("factor").map_elements(
                lambda f: factor_expr_map.get(f, "unknown"),
                return_dtype=pl.String
            ).alias("expression")
        ).sort("sharpe", descending=True)
    )

    # --- 5. 结果展示 ---
    logger.success("✅ 评估任务完成，结果概览:")
    # 强制显示所有行，不进行截断
    with pl.Config(tbl_rows=100, tbl_width_chars=200):
        print(df_result)


if __name__ == "__main__":
    run_batch_factor_test()
