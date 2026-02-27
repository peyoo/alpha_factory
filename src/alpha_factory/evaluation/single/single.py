"""
单因子深度分析工具集
包含 IC 计算、分层收益分析、衰减与换手率等功能

"""

from typing import Literal, Union

import polars as pl
from loguru import logger

from alpha_factory.evaluation.batch.ic_summary import batch_ic_summary
from alpha_factory.evaluation.single.quantile_metric import single_calc_quantile_metrics
from alpha_factory.evaluation.single.turnover_decay import single_calc_decay_turnover
from alpha_factory.utils.schema import F


def single_factor_alpha_analysis(
    df: Union[pl.DataFrame, pl.LazyFrame],
    factor_col: str,
    ret_col: str = F.LABEL_FOR_RET,
    date_col: str = F.DATE,
    asset_col: str = F.ASSET,
    pool_mask_col: str = F.POOL_MASK,
    mode: Literal["long_only", "long_short", "active"] = "long_only",
    n_bins: int = 5,
    period: int = 1,
    cost: float = 0.003,  # 默认单边费率（如印花税+佣金）
) -> dict:
    """
    【工业级】单因子全能体检报告：
    集成信号衰减、自相关性换手估算、扣费分层回测。
    """

    # 1. 信号衰减与换手率估算 (核心：先算稳定性)
    # 返回包含 ic_lags, autocorr, est_daily_turnover 的字典
    logger.info("🔍 正在计算因子信号衰减与换手率估算...")
    decay_stats = single_calc_decay_turnover(
        df, factor_col, ret_col, date_col, asset_col
    )
    logger.info(
        f"    > 估算日均换手率: {decay_stats['est_daily_turnover']:.2%} (自相关: {decay_stats['autocorr']:.3f})"
    )
    est_turnover = decay_stats["est_daily_turnover"]

    logger.info("🔍 正在计算因子预测效力指标 (IC Summary)...")
    # 2. 基础 IC 统计 (预测效力)
    ic_summary = batch_ic_summary(
        df, factors=f"^{factor_col}$", label_for_ic=ret_col, date_col=date_col
    )
    ic_mean = ic_summary["ic_mean"][0]
    logger.info(f"    > IC 均值: {ic_mean:.4f}, ICIR: {ic_summary['ic_ir'][0]:.4f}")

    # 3. 分层收益与实盘风险指标 (传入估算的 est_turnover 进行扣费)
    quantile_res = single_calc_quantile_metrics(
        df,
        factor_col,
        ret_col,
        date_col=date_col,
        asset_col=asset_col,
        pool_mask_col=pool_mask_col,
        mode=mode,
        n_bins=n_bins,
        period=period,
        cost=cost,
        est_turnover=est_turnover,  # 自动关联换手
        direction=1 if ic_mean > 0 else -1,  # 根据信号方向调整多空逻辑
    )

    m = quantile_res["metrics"]
    nav_series = quantile_res["series"]

    # --- 开始打印全量解释报告 ---
    print(f"\n{'#' * 30} 因子体检报告: {factor_col} {'#' * 30}")

    # --- 第一部分：预测效力 ---
    print("\n【1. 预测效力 - 衡量因子捕捉收益的相关性】")
    ic_val = ic_summary["ic_mean"][0]
    icir_val = ic_summary["ic_ir"][0]
    print(f"  > IC 均值: {ic_val:.4f}")
    print("    [解释]: 因子值与下期收益的相关系数。>0.02代表有预测力，值越大方向越准。")
    print(f"  > ICIR: {icir_val:.4f}")
    print("    [解释]: IC均值/IC标准差。衡量稳定性，>0.5代表信号稳健。")

    # --- 第二部分：实盘表现 ---
    print("\n【2. 实盘表现 - 模拟真实交易扣费后的收益】")
    print(f"  > 净年化收益: {m['annual_return']:.2%}")
    print("    [解释]: 考虑调仓周期和基于自相关性估算的换手扣费后的年化。")
    print(f"  > 净夏普比率: {m['sharpe_ratio']:.2f}")
    print(f"  > 最大回撤: {m['max_drawdown']:.2%}")

    # --- 第三部分：执行成本 ---
    print("\n【3. 执行成本 - 衡量因子在实盘中落地的难易度】")
    print(f"  > 估算日均换手率: {est_turnover:.2%}")
    print(
        f"    [解释]: 基于因子秩自相关性(AC={decay_stats['autocorr']:.3f})推导出的每日头寸变动。"
    )
    print(f"  > 调仓周期: {period} 交易日")
    print(f"  > 摩擦成本系数: {cost * 10000:.1f} bps (基点)")

    # --- 第四部分：逻辑健壮性 ---
    print("\n【4. 逻辑健壮性 - 检验因子赚钱的底层逻辑】")
    mono = m.get("monotonicity")
    smooth = m.get("smoothness_index")

    def _fmt(x):
        try:
            return f"{x:.2f}"
        except Exception:
            return str(x)

    print(f"  > 收益单调性: {_fmt(mono)}")
    # 解释单调性
    if mono is None or (isinstance(mono, float) and (mono != mono)):
        print("    [解释]: 单调性未定义（样本不足或计算异常）。")
    else:
        abs_m = abs(mono)
        if abs_m >= 0.8:
            msg = "非常好：分层收益随因子排序高度单调，因子区分度强。"
        elif abs_m >= 0.5:
            msg = "较好：存在稳定单调关系，通常为可交易信号（视波动/换手而定）。"
        elif abs_m >= 0.2:
            msg = "一般：单调性弱，可能依赖少数样本或特定区间表现好。"
        else:
            msg = "较差：未见明显单调关系，因子在区分收益上效力有限。"

        # 方向性提示
        if mono < 0:
            dir_msg = "（负向单调：因子值越小越好，注意在回测中调整方向）"
        else:
            dir_msg = ""

        print(f"    [解释]: {msg} {dir_msg}")

    print(f"  > 分层平滑度: {_fmt(smooth)}")
    # 解释平滑度
    if smooth is None or (isinstance(smooth, float) and (smooth != smooth)):
        print("    [解释]: 平滑度未定义（样本不足或计算异常）。")
    else:
        if smooth >= 0.8:
            s_msg = "非常平滑：各分层收益间距稳定，信号鲁棒性高。"
        elif smooth >= 0.5:
            s_msg = "较为平滑：分层间距有一定稳定性，整体可信度良好。"
        else:
            s_msg = "不平滑：分层收益波动或间距不稳定，可能受噪声或极端值影响。"

        print(
            f"    [解释]: {s_msg} 若不平滑，请检查是否存在少数极端样本、资产集中或行业偏移。"
        )

    # 基于两项综合建议
    try:
        score = 0.0
        if isinstance(mono, (int, float)) and mono == mono:
            score += min(1.0, abs(mono))
        if isinstance(smooth, (int, float)) and smooth == smooth:
            score += min(1.0, smooth)

        if score >= 1.6:
            overall = "优秀：因子具有明确且稳定的分层能力，适合中低频实盘化。"
        elif score >= 1.0:
            overall = "良好：因子有可交易信号，但需注意换手和成本控制。"
        elif score >= 0.5:
            overall = "一般：因子存在一定信息，但需进一步净化或与其它因子组合使用。"
        else:
            overall = "较差：建议回溯数据源、处理异常值或放弃该因子。"

        print(f"    [综合评估]: {overall}")
    except Exception:
        pass

    # --- 第五部分：信号衰减 ---
    print("\n【5. 信号衰减 - 衡量因子的“保鲜期”】")
    lags = decay_stats["ic_lags"]
    # 避免除以 0，且 lag0 通常是当期 IC
    lag1_val = lags[1] if len(lags) > 1 else 1e-9
    lag5_val = lags[5] if len(lags) > 5 else 0.0
    retention = (lag5_val / lag1_val) if lag1_val != 0 else 0.0
    print(f"  > 信号留存率 (Lag5/Lag1): {retention:.1%}")
    print("    [解释]: 5天后预测能力剩下的比例。若<20%，说明该因子必须高频调仓。")

    # --- 样本统计 ---
    print("\n【6. 样本统计】")
    print(f"  > 每层平均样本数: {m['avg_count_per_bin']:.1f}")

    print(f"\n{'#' * 78}\n")
    logger.info("✅ 因子体检报告生成完毕。")

    return {
        "summary": ic_summary,
        "metrics": m,
        "decay": decay_stats,
        "nav": nav_series,
    }
