from typing import Union, List, Literal
import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger

from alpha_factory.utils.schema import F


def batch_full_metrics(
    df: Union[pl.DataFrame, pl.LazyFrame],
    factors: Union[str, List[str]] = r"^factor_.*",
    label_ic_col: str = F.LABEL_FOR_IC,
    label_ret_col: str = F.LABEL_FOR_RET,
    date_col: str = F.DATE,
    asset_col: str = F.ASSET,
    pool_mask_col: str = F.POOL_MASK,
    n_bins: int = 10,
    mode: Literal["long_only", "long_short", "active"] = "long_only",
    annual_days: int = 252,
    fee: float = 0.0025,
) -> pl.DataFrame:
    """
    批量因子评估（基于目标股票池）

    Args:
        df: 输入数据框，包含因子列、标签列及股票池掩码列
        pool_mask_col: 动态股票池掩码列（True = 在池内）
            示例：_POOL_MASK_（包含流通市值过滤 + 停牌过滤）
            每日动态变化，反映实际可投资范围
        factors: 因子列选择器（正则表达式或名称列表）
        label_ic_col: 用于计算 IC 的标签列
        label_ret_col: 用于计算收益率的标签列
        date_col: 日期列名称
        asset_col: 资产列名称
        n_bins: 分桶数量（用于多分位信号生成）
        mode: 评估模式
            - 'long_only': 仅多头（top bin）持仓
            - 'long_short': 多头（top bin）减空头（bottom bin）
            - 'active': 多头收益相对市场平均的超额收益
        annual_days: 年化交易日天数（用于年化收益计算）
        fee: 单边交易费用率（换手成本 = turnover * fee * 2）

    Returns:
        pl.DataFrame: 因子评估结果，按 sharpe 降序排列，Schema 如下：
        | 列名          | 类型    | 说明                              |
        | :------------ | :------ | :-------------------------------- |
        | factor        | String  | 因子名称 (e.g., 'factor_0')      |
        | ic_mean       | Float64 | 每日 IC 的算术平均值              |
        | ic_mean_abs   | Float64 | IC 均值绝对值（常用于进化目标）   |
        | ic_ir         | Float64 | IC 信息比率 (ic_mean / ic_std)    |
        | ic_ir_abs     | Float64 | IC IR 的绝对值                    |
        | ann_ret       | Float64 | 年化收益率                        |
        | sharpe        | Float64 | 夏普比率                          |
        | turnover_est  | Float64 | 换手率估计                        |
        | direction     | Int32   | 因子方向 (1=正向, -1=反向)        |

    Notes:
        - 所有指标均基于目标股票池内计算，确保与实际可交易范围一致
        - Rank / 分桶在过滤后的池内进行，非全市场
        - 换手率分母加保护，防止每日股票数极少时除零
        - long_short 模式下，收益 = top_bin_ret - btm_bin_ret
        - ic_mean 使用显式 None 判断，避免 `or 0.0` 误将真实 0.0 替换
    """
    logger.info(
        f"▶ batch_full_metrics 开始 | mode={mode} | n_bins={n_bins} "
        f"| annual_days={annual_days} | fee={fee}"
    )

    # ─────────────────────────────────────────────
    # Step 1: 转为 LazyFrame 并按 [asset, date] 排序
    #   排序目的：保证 shift(1).over(asset) 的时序正确性
    # ─────────────────────────────────────────────
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    lf = lf.sort([asset_col, date_col])
    logger.debug(f"  已转为 LazyFrame 并按 [{asset_col}, {date_col}] 排序")

    # ─────────────────────────────────────────────
    # Step 2: 过滤到目标股票池（小微盘 + 可交易）
    #   pool_mask_col=True 表示该资产在当日可投资范围内
    #   必须最先过滤，确保后续 Rank/分桶基于真实可投资范围
    # ─────────────────────────────────────────────
    schema_names = lf.collect_schema().names()
    if pool_mask_col in schema_names:
        lf = lf.filter(pl.col(pool_mask_col))
        logger.info(f"  ✓ 已应用股票池掩码列: '{pool_mask_col}'")
    else:
        logger.warning(f"  ⚠️ 未找到池掩码列 '{pool_mask_col}'，将使用全市场数据评估")

    # ─────────────────────────────────────────────
    # Step 3: 解析因子列名
    #   支持正则（如 r"^factor_.*"）或显式列名列表
    # ─────────────────────────────────────────────
    f_selector = (
        cs.matches(factors) if isinstance(factors, str) else cs.by_name(factors)
    )
    factor_cols = lf.select(f_selector).collect_schema().names()

    if not factor_cols:
        logger.error(f"  ✗ 未匹配到任何因子列，选择器: {factors}，返回空 DataFrame")
        return pl.DataFrame()

    logger.info(f"  ✓ 匹配到 {len(factor_cols)} 个因子列: {factor_cols}")

    # ─────────────────────────────────────────────
    # Step 4: 计算每日截面内的因子 Rank
    #   - rank() 默认升序（小值 rank 小）
    #   - over(date_col) 表示按日截面分组，即每日独立排名
    #   - 后续 top/btm 分桶均基于此 rank 值
    # ─────────────────────────────────────────────
    logger.debug(f"  计算因子 Rank（截面内，窗口={date_col}）...")
    lf = lf.with_columns(
        [pl.col(f).rank().over(date_col).alias(f"{f}_rank") for f in factor_cols]
    )

    # ─────────────────────────────────────────────
    # Step 5: 生成 top/btm 分桶标志
    #   - is_top: rank 位于前 1/n_bins（最高分位，因子值最大）
    #   - is_btm: rank 位于后 1/n_bins（最低分位，因子值最小）
    #   - over(date_col) 确保分母 len() 为当日实际股票数
    # ─────────────────────────────────────────────
    logger.debug(f"  生成 top/btm 分桶标志（n_bins={n_bins}）...")
    daily_pool_size = pl.len().over(date_col)
    lf = lf.with_columns(
        [
            expr
            for f in factor_cols
            for expr in [
                (pl.col(f"{f}_rank") > (daily_pool_size * (n_bins - 1) / n_bins)).alias(
                    f"{f}_is_top"
                ),
                (pl.col(f"{f}_rank") <= (daily_pool_size / n_bins)).alias(
                    f"{f}_is_btm"
                ),
            ]
        ]
    )

    # ─────────────────────────────────────────────
    # Step 6: 生成持仓信号与换手信号
    #   - sig_top/sig_btm：T-1 日的 top/btm 标志（shift(1)），
    #     表示"昨日已持仓"，用于 T 日的收益归因（避免前视偏差）
    #   - buy_top/buy_btm：今日进入 top/btm 且昨日未持仓，
    #     用于估计新建仓换手率
    #   - fill_null(False)：首日无历史信号，默认为未持仓
    # ─────────────────────────────────────────────
    logger.debug("  生成持仓信号（shift(1) 避免前视偏差）与换手信号...")
    lf = lf.with_columns(
        [
            expr
            for f in factor_cols
            for expr in [
                # 昨日是否在 top 分桶（T 日持仓标志）
                pl.col(f"{f}_is_top")
                .shift(1)
                .over(asset_col)
                .fill_null(False)
                .alias(f"{f}_sig_top"),
                # 昨日是否在 btm 分桶（T 日持仓标志）
                pl.col(f"{f}_is_btm")
                .shift(1)
                .over(asset_col)
                .fill_null(False)
                .alias(f"{f}_sig_btm"),
                # 今日新进入 top（换手信号）
                (
                    pl.col(f"{f}_is_top")
                    & ~pl.col(f"{f}_is_top").shift(1).over(asset_col).fill_null(False)
                ).alias(f"{f}_buy_top"),
                # 今日新进入 btm（换手信号）
                (
                    pl.col(f"{f}_is_btm")
                    & ~pl.col(f"{f}_is_btm").shift(1).over(asset_col).fill_null(False)
                ).alias(f"{f}_buy_btm"),
            ]
        ]
    )

    # ─────────────────────────────────────────────
    # Step 7: 日度截面聚合
    #   每日聚合以下指标：
    #   - market_avg:    当日全池平均收益率（active 模式基准）
    #   - {f}_ic:        因子与 label_ic_col 的 Spearman 相关系数
    #   - {f}_top_ret:   top 分桶持仓股票（sig_top=True）的平均收益
    #   - {f}_btm_ret:   btm 分桶持仓股票（sig_btm=True）的平均收益
    #   - {f}_to_top:    top 分桶换手率（新进入占比），分母加 clip 防除零
    #   - {f}_to_btm:    btm 分桶换手率
    # ─────────────────────────────────────────────
    logger.info("  执行日度截面聚合（group_by date）...")
    daily_stats = (
        lf.group_by(date_col)
        .agg(
            [
                # 当日全池平均收益（active 模式超额收益基准）
                pl.col(label_ret_col).mean().alias("market_avg"),
                # Spearman IC：衡量因子截面排序与未来收益的相关性
                *[
                    pl.corr(f, label_ic_col, method="spearman").alias(f"{f}_ic")
                    for f in factor_cols
                ],
                # top 分桶（T-1 信号）当日平均收益
                *[
                    pl.col(label_ret_col)
                    .filter(pl.col(f"{f}_sig_top"))
                    .mean()
                    .alias(f"{f}_top_ret")
                    for f in factor_cols
                ],
                # btm 分桶（T-1 信号）当日平均收益
                *[
                    pl.col(label_ret_col)
                    .filter(pl.col(f"{f}_sig_btm"))
                    .mean()
                    .alias(f"{f}_btm_ret")
                    for f in factor_cols
                ],
                # top 换手率：新进入 top 的股票数 / 理论持仓数（池大小 / n_bins）
                # clip(lower_bound=1) 防止极少股票日除零
                *[
                    (
                        pl.col(f"{f}_buy_top").sum()
                        / pl.len().truediv(n_bins).clip(lower_bound=1)
                    ).alias(f"{f}_to_top")
                    for f in factor_cols
                ],
                # btm 换手率（同上）
                *[
                    (
                        pl.col(f"{f}_buy_btm").sum()
                        / pl.len().truediv(n_bins).clip(lower_bound=1)
                    ).alias(f"{f}_to_btm")
                    for f in factor_cols
                ],
            ]
        )
        .sort(date_col)
        .collect()
    )

    total_days = daily_stats.height
    logger.info(f"  ✓ 日度聚合完成，共 {total_days} 个交易日")

    # ─────────────────────────────────────────────
    # Step 8: 逐因子汇总统计
    #   对每个因子计算以下汇总指标：
    #   - ic_mean / ic_std → ic_ir（信息比率）
    #   - direction：由 ic_mean 符号决定，正向因子用 top，反向用 btm
    #   - mode 决定收益序列构建方式：
    #       long_only  → 单边 top/btm 净收益
    #       long_short → top - btm 多空收益
    #       active     → top 超额收益（相对全池均值）
    #   - 扣费后 NAV → 年化收益率 / 波动率 / 夏普比率
    # ─────────────────────────────────────────────
    logger.info(f"  计算各因子汇总统计（mode='{mode}'）...")
    market_avg_np = daily_stats.get_column("market_avg").fill_null(0.0).to_numpy()
    final_results = []

    for f in factor_cols:
        ic_series = daily_stats.get_column(f"{f}_ic")

        # 使用显式 None 判断，避免 `ic_mean or 0.0` 将真实 0.0 错误替换
        raw_ic_mean = ic_series.mean()
        ic_mean = float(raw_ic_mean) if raw_ic_mean is not None else 0.0
        ic_mean_abs = abs(ic_mean)

        raw_ic_std = ic_series.std()
        ic_std = float(raw_ic_std) if raw_ic_std is not None else 1e-9
        ic_std = ic_std if ic_std > 0 else 1e-9  # 防止 std=0 时 IR 无穷大

        # 因子方向：ic_mean > 0 → 正向（top 为多头），否则反向（btm 为多头）
        direction = 1 if ic_mean >= 0 else -1
        # logger.debug(
        #     f"    [{f}] ic_mean={ic_mean:.4f} | ic_std={ic_std:.4f} "
        #     f"| direction={direction}"
        # )

        if mode == "long_short":
            # 多空对冲：top 收益 - btm 收益，换手为两侧均值
            top_ret = daily_stats.get_column(f"{f}_top_ret").fill_null(0.0).to_numpy()
            btm_ret = daily_stats.get_column(f"{f}_btm_ret").fill_null(0.0).to_numpy()
            raw_ret = top_ret - btm_ret

            raw_to_top = daily_stats.get_column(f"{f}_to_top").mean()
            raw_to_btm = daily_stats.get_column(f"{f}_to_btm").mean()
            turnover_val = (
                (float(raw_to_top) if raw_to_top is not None else 0.0)
                + (float(raw_to_btm) if raw_to_btm is not None else 0.0)
            ) / 2
            # logger.debug(
            #     f"    [{f}] long_short 换手率均值: top={raw_to_top:.4f}, "
            #     f"btm={raw_to_btm:.4f}"
            # )

        else:
            # long_only 或 active：根据 direction 选择多头方向
            if direction == 1:
                raw_ret = (
                    daily_stats.get_column(f"{f}_top_ret").fill_null(0.0).to_numpy()
                )
                raw_to = daily_stats.get_column(f"{f}_to_top").mean()
            else:
                raw_ret = (
                    daily_stats.get_column(f"{f}_btm_ret").fill_null(0.0).to_numpy()
                )
                raw_to = daily_stats.get_column(f"{f}_to_btm").mean()
            turnover_val = float(raw_to) if raw_to is not None else 0.0

            if mode == "active":
                # 超额收益 = 多头收益 - 全池市场平均收益
                raw_ret = raw_ret - market_avg_np
                # logger.debug(f"    [{f}] active 模式：已扣除市场均值")

        # 扣除双边交易费用（buy + sell 各一次）
        # net_daily_ret[i] = raw_ret[i] - turnover * fee * 2
        net_daily_ret = raw_ret - (turnover_val * fee * 2)
        # logger.debug(
        #     f"    [{f}] 换手率估计={turnover_val:.4f} | "
        #     f"单日费用={turnover_val * fee * 2:.6f}"
        # )

        # 累乘 NAV（净资产价值序列）
        nav = np.cumprod(1.0 + net_daily_ret)
        total_ret = nav[-1] - 1 if len(nav) > 0 else 0.0

        # 年化收益：(1 + total_ret)^(annual_days / total_days) - 1
        ann_ret = (
            (1 + total_ret) ** (annual_days / total_days) - 1 if total_days > 0 else 0.0
        )
        # 年化波动率：日度标准差 * sqrt(annual_days)
        vol = np.std(net_daily_ret) * np.sqrt(annual_days)
        # 夏普比率：年化收益 / 年化波动率，加 1e-9 防除零
        sharpe = ann_ret / (vol + 1e-9)

        # logger.debug(
        #     f"    [{f}] ann_ret={ann_ret:.4f} | vol={vol:.4f} | sharpe={sharpe:.4f}"
        # )

        final_results.append(
            {
                "factor": f,
                "ic_mean": ic_mean,
                "ic_mean_abs": ic_mean_abs,
                "ic_ir": ic_mean / ic_std,
                "ic_ir_abs": ic_mean_abs / ic_std,
                "ann_ret": ann_ret,
                "sharpe": sharpe,
                "turnover_est": turnover_val,
                "direction": direction,
            }
        )

    result_df = pl.DataFrame(final_results).sort("sharpe", descending=True)
    logger.info(
        f"✅ batch_full_metrics 完成 | 共评估 {len(factor_cols)} 个因子 "
        f"| 最优因子: {result_df['factor'][0]} "
        f"(sharpe={result_df['sharpe'][0]:.4f})"
    )
    return result_df
