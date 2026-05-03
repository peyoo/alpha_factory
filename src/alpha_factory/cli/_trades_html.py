"""_trades_html.py — trades 命令的 HTML 报告生成函数。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from rich.console import Console

from alpha_factory.cli._trades_viz import (
    _detect_shap_threshold,
    _shap_decision_plots,
    _shap_dependence_grid,
    _shap_heatmap_by_time,
    _shap_interaction_plots,
    _shap_to_base64,
)
from alpha_factory.config.base import settings
from alpha_factory.config.strategy import StrategyConfig

console = Console()


def _build_diagnosis_html(
    filter_result: dict,
    rank_result: dict | None,
    bad_threshold: float,
    mode: str = "both",
) -> str:
    """根据 SHAP 均值自动生成中文诊断卡片 HTML。"""
    try:
        import numpy as np

        f_shap = filter_result.get("shap_values")
        r_shap = rank_result.get("shap_values") if rank_result is not None else None
        f_names = filter_result.get("feature_names", [])
        r_names = (
            rank_result.get("feature_names", []) if rank_result is not None else []
        )

        if f_shap is None:
            return ""

        f_mean = np.abs(f_shap).mean(axis=0)

        # 坏交易最危险因子：filter top-3
        f_top_idx = np.argsort(f_mean)[::-1][:3]
        f_top = [(f_names[i], float(f_mean[i])) for i in f_top_idx if i < len(f_names)]

        # 双确认（同时出现在 filter top-5 和 rank top-5）
        f_top5_names = {
            f_names[i] for i in np.argsort(f_mean)[::-1][:5] if i < len(f_names)
        }
        r_top: list[tuple[str, float]] = []
        dual: list[str] = []
        if r_shap is not None:
            r_mean = np.abs(r_shap).mean(axis=0)
            # 高收益核心因子：rank top-3
            r_top_idx = np.argsort(r_mean)[::-1][:3]
            r_top = [
                (r_names[i], float(r_mean[i])) for i in r_top_idx if i < len(r_names)
            ]
            r_top5_names = {
                r_names[i] for i in np.argsort(r_mean)[::-1][:5] if i < len(r_names)
            }
            dual = sorted(f_top5_names & r_top5_names)

        def card(icon, title, body, color):
            return f"""<div style="background:{color};border-radius:8px;padding:12px 16px;margin:8px 0;border-left:4px solid #1a5276">
<b style="font-size:1.05em">{icon} {title}</b>
<div style="margin-top:6px;line-height:1.7;font-size:.92em">{body}</div>
</div>"""

        # 卡片1：防御
        shield_rows = "".join(
            f"<li><b>{n}</b> (|SHAP| 均值={v:.4f})"
            + (
                lambda t: (
                    f"，非线性阈值 ≈ <b>{t:.3g}</b>，建议在该阈值处过滤"
                    if t is not None
                    else "，暂未检测到明显非线性阈值"
                )
            )(
                _detect_shap_threshold(
                    filter_result.get("X_full", np.zeros((1, len(f_names))))[
                        :, f_names.index(n)
                    ]
                    if n in f_names
                    else np.array([0]),
                    f_shap[:, f_names.index(n)] if n in f_names else np.array([0]),
                )
                if f_names
                else None
            )
            + "</li>"
            for n, v in f_top
        )
        card1 = card(
            "🛡",
            "防御盾牌 — 坏交易风险因子",
            f"以下因子对 <b>pnl_ret ≤ {bad_threshold}</b> 的坏交易贡献最大，建议加入 <code>pool_mask</code> / <code>not_buy_able</code>：<ul>{shield_rows}</ul>",
            "#fff3f3",
        )

        # 卡片2：进攻
        atk_rows = "".join(
            f"<li><b>{n}</b> (|SHAP| 均值={v:.4f})</li>" for n, v in r_top
        )
        card2 = card(
            "⚔",
            "进攻矛头 — 高收益驱动因子",
            f"以下因子对高收益 <b>pnl_ret</b> 贡献最大，建议加入/增权 <code>ranks</code>：<ul>{atk_rows}</ul>",
            "#f3fff3",
        )

        # 卡片3：双确认
        if dual:
            dual_items = "".join(f"<li><b>{n}</b></li>" for n in dual)
            card3 = card(
                "🔥",
                "双确认因子 — 排雷选优均显著",
                f"以下因子同时出现在 filter top-5 和 rank top-5，表明其对交易结果有强烈的非线性影响，建议重点审查：<ul>{dual_items}</ul>",
                "#fffbf0",
            )
        else:
            card3 = card(
                "🔥",
                "双确认因子",
                "未检测到同时出现在 filter/rank top-5 的因子，各维度独立发挥作用。",
                "#fffbf0",
            )

        cards_html = "\n".join(
            [
                *((card1,) if mode in ("filter", "both") else ()),
                *((card2,) if mode in ("rank", "both") else ()),
                card3,
            ]
        )
        return f"""<div class="section" id="diagnosis">
<h2>🩺 自动诊断卡片</h2>
<p class="hint">以下诊断由 SHAP 值自动生成，供策略调优参考。</p>
{cards_html}
</div>"""
    except Exception:  # noqa: BLE001
        return ""


def _generate_html_report(
    cfg: StrategyConfig,
    trades_feat: pl.DataFrame,
    bad_threshold: float,
    filter_result: dict,
    rank_result: dict | None,
    full_diag: bool = False,
) -> tuple[Path, Path | None]:
    import numpy as np

    f_names = filter_result.get("feature_names", [])
    r_names = rank_result.get("feature_names", []) if rank_result is not None else []

    # ── Section 1：Global Landscape（beeswarm + bar）──────────────────────
    console.print("[dim]  [1/5] Global landscape (beeswarm + bar)...[/dim]")
    f_bee = _shap_to_base64(
        filter_result["shap_values"],
        filter_result["X_test_np"],
        f_names,
        "beeswarm",
        "filter: SHAP beeswarm",
    )
    f_bar = _shap_to_base64(
        filter_result["shap_values"],
        filter_result["X_test_np"],
        f_names,
        "bar",
        "filter: Feature Importance",
    )
    r_bee = (
        _shap_to_base64(
            rank_result["shap_values"],
            rank_result["X_test_np"],
            r_names,
            "beeswarm",
            "rank: SHAP beeswarm",
        )
        if rank_result is not None
        else ""
    )
    r_bar = (
        _shap_to_base64(
            rank_result["shap_values"],
            rank_result["X_test_np"],
            r_names,
            "bar",
            "rank: Feature Importance",
        )
        if rank_result is not None
        else ""
    )

    # ── Section 2：Dependence Grid ────────────────────────────────────────
    console.print("[dim]  [2/5] Dependence plots...[/dim]")
    f_dep_imgs: list[str] = []
    r_dep_imgs: list[str] = []
    if filter_result.get("shap_values") is not None:
        f_dep_imgs = _shap_dependence_grid(
            filter_result["shap_values"], filter_result["X_test_np"], f_names
        )
    if rank_result is not None and rank_result.get("shap_values") is not None:
        r_dep_imgs = _shap_dependence_grid(
            rank_result["shap_values"], rank_result["X_test_np"], r_names
        )

    # ── Section 3：Interaction（仅 full_diag）────────────────────────────
    f_inter_img = ""
    r_inter_img = ""
    if full_diag:
        console.print("[dim]  [3/5] SHAP Interaction（最慢，请稍候...）[/dim]")
        X_full_f = filter_result.get("X_full")
        if X_full_f is not None and filter_result.get("model") is not None:
            console.print("[dim]        → filter 模型...[/dim]")
            f_inter_img = _shap_interaction_plots(
                filter_result["model"], X_full_f, f_names
            )
        if rank_result is not None:
            X_full_r = rank_result.get("X_full")
            if X_full_r is not None and rank_result.get("model") is not None:
                console.print("[dim]        → rank 模型...[/dim]")
                r_inter_img = _shap_interaction_plots(
                    rank_result["model"], X_full_r, r_names
                )
    else:
        console.print("[dim]  [3/5] Interaction 已跳过（使用 --full-diag 开启）[/dim]")

    # ── Section 4：Time Heatmap ───────────────────────────────────────────
    console.print("[dim]  [4/5] Time heatmap...[/dim]")
    f_heat_img = ""
    r_heat_img = ""
    if (
        filter_result.get("shap_values") is not None
        and filter_result.get("buy_dates_full") is not None
    ):
        f_heat_img = _shap_heatmap_by_time(
            filter_result["shap_values"],
            filter_result["X_test_np"],
            f_names,
            filter_result["buy_dates_full"][
                len(filter_result["buy_dates_full"])
                - len(filter_result["shap_values"]) :
            ],
        )
    if (
        rank_result is not None
        and rank_result.get("shap_values") is not None
        and rank_result.get("buy_dates_full") is not None
    ):
        r_heat_img = _shap_heatmap_by_time(
            rank_result["shap_values"],
            rank_result["X_test_np"],
            r_names,
            rank_result["buy_dates_full"][
                len(rank_result["buy_dates_full"]) - len(rank_result["shap_values"]) :
            ],
        )

    # ── Section 5：Decision Plot（worst-20）──────────────────────────────
    console.print("[dim]  [5/5] Decision plots...[/dim]")
    f_dec_img = ""
    r_dec_img = ""
    worst_n = 20
    if (
        filter_result.get("shap_values") is not None
        and filter_result.get("pnl_test") is not None
        and filter_result.get("shap_explainer") is not None
    ):
        pnl_t = filter_result["pnl_test"]
        sv = filter_result["shap_values"]
        xt = filter_result["X_test_np"]
        if len(pnl_t) >= worst_n:
            worst_idx = np.argsort(pnl_t)[:worst_n]
        else:
            worst_idx = np.argsort(pnl_t)
        f_dec_img = _shap_decision_plots(
            filter_result["shap_explainer"],
            sv[worst_idx],
            xt[worst_idx],
            f_names,
        )
    if (
        rank_result is not None
        and rank_result.get("shap_values") is not None
        and rank_result.get("pnl_test") is not None
        and rank_result.get("shap_explainer") is not None
    ):
        pnl_t = rank_result["pnl_test"]
        sv = rank_result["shap_values"]
        xt = rank_result["X_test_np"]
        if len(pnl_t) >= worst_n:
            worst_idx = np.argsort(pnl_t)[:worst_n]
        else:
            worst_idx = np.argsort(pnl_t)
        r_dec_img = _shap_decision_plots(
            rank_result["shap_explainer"],
            sv[worst_idx],
            xt[worst_idx],
            r_names,
        )

    # ── 公共辅助 ─────────────────────────────────────────────────────────
    def img_tag(b64: str, alt: str) -> str:
        if not b64:
            return f"<p style='color:gray'>[{alt} 图表生成失败]</p>"
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;margin:8px 0" alt="{alt}"/>'

    pnl = trades_feat["pnl_ret"]
    n = len(pnl)
    n_bad = int((pnl <= bad_threshold).sum())
    win_rate = float((pnl > 0).sum()) / n if n > 0 else float("nan")

    def fmt(v, fmt_str=".4f"):
        return f"{v:{fmt_str}}" if v == v else "N/A"

    def dep_section(imgs: list[str]) -> str:
        if not imgs:
            return ""
        inner = "\n".join(
            f'<div style="display:inline-block;margin:4px">{img_tag(b, f"dep {i}")}</div>'
            for i, b in enumerate(imgs)
        )
        return (
            '<div style="margin-top:12px">'
            "<h3>Dependence Plots</h3>"
            '<p class="hint">散点 X=特征值，Y=SHAP 值，颜色=最高交互特征。红虚线为非线性阈值（若检测到）。</p>'
            f'<div style="display:flex;flex-wrap:wrap">{inner}</div>'
            "</div>"
        )

    _CSS = (
        "  body {font-family:system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:20px;color:#222}\n"
        "  h1 {color:#1a5276;border-bottom:2px solid #1a5276;padding-bottom:8px}\n"
        "  h2 {color:#1f618d;margin-top:32px}\n"
        "  h3 {color:#333;margin-top:16px}\n"
        "  table {border-collapse:collapse;width:100%;margin:12px 0}\n"
        "  th,td {border:1px solid #ddd;padding:8px 12px;text-align:left}\n"
        "  th {background:#2980b9;color:#fff}\n"
        "  tr:nth-child(even) {background:#f5f5f5}\n"
        "  .section {background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:16px;margin:16px 0}\n"
        "  .badge-filter {background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-size:.85em}\n"
        "  .badge-rank {background:#27ae60;color:#fff;padding:2px 8px;border-radius:4px;font-size:.85em}\n"
        "  .hint {color:#555;font-size:.9em;line-height:1.6}\n"
    )

    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_meta = (
        f"buy_rank={cfg.buy_rank} &nbsp;|&nbsp; sell_rank={cfg.sell_rank} &nbsp;|&nbsp; "
        f"坏交易阈值={bad_threshold}"
        + ("&nbsp;|&nbsp;<b>full-diag 模式</b>" if full_diag else "")
    )
    stats_section_html = (
        f'<div class="section"><h2>交易统计</h2>'
        f"<table><tr><th>指标</th><th>数值</th></tr>"
        f"<tr><td>交易总数</td><td>{n}</td></tr>"
        f"<tr><td>坏交易数（pnl_ret ≤ {bad_threshold}）</td><td>{n_bad}</td></tr>"
        f"<tr><td>胜率（pnl_ret &gt; 0）</td><td>{win_rate:.2%}</td></tr>"
        f"<tr><td>平均收益</td><td>{fmt(float(pnl.mean() or 0.0))}</td></tr>"
        f"<tr><td>平均持仓天数</td><td>"
        f"{fmt(float(trades_feat['hold_days'].mean() or 0.0), '.1f')}</td></tr>"
        "</table></div>"
    )

    console.print("[dim]  生成诊断卡片...[/dim]")

    # ── Filter 报告 ───────────────────────────────────────────────────────
    f_diag_html = _build_diagnosis_html(
        filter_result, rank_result, bad_threshold, mode="filter"
    )
    f_inter_section = (
        f'<div class="section"><h2>⚡ Section 3 — 交互与协同效应'
        f' <small style="font-size:.7em;color:#888">（--full-diag）</small></h2>'
        f'<p class="hint">SHAP Interaction Value 热图（左）+ 最强交互对散点图（右）。</p>'
        f"{img_tag(f_inter_img, 'filter interaction')}</div>"
        if full_diag
        else ""
    )
    filter_html = (
        f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"/>'
        f"<title>Filter 分析 — 排雷 · {cfg.name}</title>"
        f"<style>{_CSS}</style></head><body>"
        f"<h1>🛡 Filter 分析 — 排雷 · {cfg.name}</h1>"
        f'<p style="color:#666">生成时间：{gen_time} &nbsp;|&nbsp; {base_meta}</p>'
        f'<p class="hint">目标：识别导致 <b>pnl_ret ≤ {bad_threshold}</b> 的坏交易特征，'
        f"建议将高风险条件加入 <code>pool_mask</code> / <code>not_buy_able</code>。</p>"
        f"{f_diag_html}{stats_section_html}"
        f'<div class="section"><h2>📊 Section 1 — Global Landscape</h2>'
        f'<p class="hint">LGBMClassifier：坏交易 label=1，AUC 越高则判别力越强。'
        f"正 SHAP → 推高坏交易概率 → 建议过滤。</p>"
        f"<table><tr><th>指标</th><th>数值</th></tr>"
        f"<tr><td>训练样本</td><td>{filter_result['n_train']}</td></tr>"
        f"<tr><td>测试样本</td><td>{filter_result['n_test']}</td></tr>"
        f"<tr><td>AUC</td><td>{fmt(filter_result.get('auc', float('nan')))}</td></tr>"
        f"<tr><td>Precision（坏=1）</td><td>{fmt(filter_result.get('precision', float('nan')))}</td></tr>"
        f"<tr><td>Recall（坏=1）</td><td>{fmt(filter_result.get('recall', float('nan')))}</td></tr>"
        f"</table>"
        f"<h4>SHAP Beeswarm</h4>{img_tag(f_bee, 'filter beeswarm')}"
        f"<h4>Feature Importance</h4>{img_tag(f_bar, 'filter importance')}</div>"
        f'<div class="section"><h2>🔍 Section 2 — 非线性边界检测（Dependence Plots）</h2>'
        f'<p class="hint">散点 X=特征值，Y=SHAP 值，颜色=最高交互特征。红虚线为非线性阈值（若检测到）。</p>'
        f"{dep_section(f_dep_imgs)}</div>"
        f"{f_inter_section}"
        f'<div class="section"><h2>🌡 Section 4 — 稳定性与风格漂移（SHAP 时间热图）</h2>'
        f'<p class="hint">行=按时间排序的交易，列=特征，颜色=SHAP 值（红=正贡献，蓝=负贡献）。</p>'
        f"{img_tag(f_heat_img, 'filter heatmap')}</div>"
        f'<div class="section"><h2>🏚 Section 5 — 诊断调试（最差 {worst_n} 笔交易 Decision Plot）</h2>'
        f'<p class="hint">Decision Plot 展示各特征如何将模型预测值从 baseline 推向最终输出。</p>'
        f"{img_tag(f_dec_img, 'filter decision')}</div>"
        "</body></html>"
    )

    # ── Rank 报告 ─────────────────────────────────────────────────────────
    r_diag_html = _build_diagnosis_html(
        filter_result, rank_result, bad_threshold, mode="rank"
    )
    r_inter_section = (
        f'<div class="section"><h2>⚡ Section 3 — 交互与协同效应'
        f' <small style="font-size:.7em;color:#888">（--full-diag）</small></h2>'
        f'<p class="hint">SHAP Interaction Value 热图（左）+ 最强交互对散点图（右）。</p>'
        f"{img_tag(r_inter_img, 'rank interaction')}</div>"
        if full_diag
        else ""
    )
    rank_html = (
        (
            f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"/>'
            f"<title>Rank 分析 — 选优 · {cfg.name}</title>"
            f"<style>{_CSS}</style></head><body>"
            f"<h1>⚔ Rank 分析 — 选优 · {cfg.name}</h1>"
            f'<p style="color:#666">生成时间：{gen_time} &nbsp;|&nbsp; {base_meta}</p>'
            f'<p class="hint">目标：识别预测高 <b>pnl_ret</b> 的特征，'
            f"建议将高权重因子加入 <code>ranks</code>。</p>"
            f"{r_diag_html}{stats_section_html}"
            f'<div class="section"><h2>📊 Section 1 — Global Landscape</h2>'
            f'<p class="hint">LGBMRegressor：目标 pnl_ret，Spearman IC 越高则预测能力越强。'
            f"正 SHAP → 推高收益 → 建议增权。</p>"
            f"<table><tr><th>指标</th><th>数值</th></tr>"
            f"<tr><td>训练样本</td><td>{rank_result['n_train']}</td></tr>"
            f"<tr><td>测试样本</td><td>{rank_result['n_test']}</td></tr>"
            f"<tr><td>Spearman IC</td><td>{fmt(rank_result.get('ic', float('nan')))}</td></tr>"
            f"<tr><td>IC p-value</td><td>{fmt(rank_result.get('ic_pval', float('nan')))}</td></tr>"
            f"<tr><td>R²</td><td>{fmt(rank_result.get('r2', float('nan')))}</td></tr>"
            f"</table>"
            f"<h4>SHAP Beeswarm</h4>{img_tag(r_bee, 'rank beeswarm')}"
            f"<h4>Feature Importance</h4>{img_tag(r_bar, 'rank importance')}</div>"
            f'<div class="section"><h2>🔍 Section 2 — 非线性边界检测（Dependence Plots）</h2>'
            f'<p class="hint">散点 X=特征值，Y=SHAP 值，颜色=最高交互特征。红虚线为非线性阈值（若检测到）。</p>'
            f"{dep_section(r_dep_imgs)}</div>"
            f"{r_inter_section}"
            f'<div class="section"><h2>🌡 Section 4 — 稳定性与风格漂移（SHAP 时间热图）</h2>'
            f'<p class="hint">行=按时间排序的交易，列=特征，颜色=SHAP 值（红=正贡献，蓝=负贡献）。</p>'
            f"{img_tag(r_heat_img, 'rank heatmap')}</div>"
            f'<div class="section"><h2>🏚 Section 5 — 诊断调试（最差 {worst_n} 笔交易 Decision Plot）</h2>'
            f'<p class="hint">Decision Plot 展示各特征如何将预测值从 baseline 推向最终输出。</p>'
            f"{img_tag(r_dec_img, 'rank decision')}</div>"
            "</body></html>"
        )
        if rank_result is not None
        else None
    )

    report_dir = Path(settings.OUTPUT_DIR) / "html_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filter_path = report_dir / f"Trades_filter_{cfg.name}_{ts}.html"
    rank_path: Path | None = None
    filter_path.write_text(filter_html, encoding="utf-8")
    if rank_html is not None:
        rank_path = report_dir / f"Trades_rank_{cfg.name}_{ts}.html"
        rank_path.write_text(rank_html, encoding="utf-8")
    return filter_path, rank_path
