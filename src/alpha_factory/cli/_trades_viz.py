"""_trades_viz.py — trades 命令的 SHAP 可视化辅助函数。"""

from __future__ import annotations

import base64
import io

_MPL_CONFIGURED = False


def _configure_mpl():
    """初始化 matplotlib：Agg 后端 + 中文字体（优先 macOS 系统字体，回退 SimHei/DejaVu）。"""
    global _MPL_CONFIGURED  # noqa: PLW0603
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not _MPL_CONFIGURED:
        plt.rcParams.update(
            {
                "font.sans-serif": [
                    "PingFang SC",
                    "Heiti SC",
                    "STHeiti",
                    "Arial Unicode MS",
                    "SimHei",
                    "DejaVu Sans",
                ],
                "axes.unicode_minus": False,
            }
        )
        _MPL_CONFIGURED = True
    return plt


def _shap_to_base64(
    shap_values,
    X_test,
    feature_names: list[str],
    plot_type: str = "beeswarm",
    title: str = "",
) -> str:
    """生成 SHAP 图并返回 base64 PNG 字符串；失败时返回空字符串。"""
    try:
        import shap
        import numpy as np

        plt = _configure_mpl()

        fig, ax = plt.subplots(figsize=(8, 5))
        plt.sca(ax)

        shap_exp = shap.Explanation(
            values=shap_values,
            data=X_test,
            feature_names=feature_names,
        )

        if plot_type == "beeswarm":
            shap.plots.beeswarm(shap_exp, show=False, max_display=len(feature_names))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
            order = np.argsort(mean_abs)[::-1]
            ax.barh(
                [feature_names[i] for i in order],
                mean_abs[order],
                color="#1f77b4",
            )
            ax.set_xlabel("mean |SHAP|")
            ax.set_title(title)
            plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# SHAP 扩展诊断辅助函数
# ---------------------------------------------------------------------------


def _detect_shap_threshold(x_vals, shap_vals) -> float | None:
    """在 20 个分位桶中找 SHAP 均值符号翻转的阈值；无翻转返回 None。"""
    try:
        import numpy as np

        x = np.asarray(x_vals, dtype=float)
        s = np.asarray(shap_vals, dtype=float)
        finite = np.isfinite(x) & np.isfinite(s)
        x, s = x[finite], s[finite]
        if len(x) < 20:
            return None
        quantiles = np.percentile(x, np.linspace(0, 100, 21))
        bucket_means = []
        bucket_centers = []
        for i in range(20):
            mask = (x >= quantiles[i]) & (x <= quantiles[i + 1])
            if mask.sum() > 0:
                bucket_means.append(float(s[mask].mean()))
                bucket_centers.append(float((quantiles[i] + quantiles[i + 1]) / 2))
        for i in range(len(bucket_means) - 1):
            if bucket_means[i] * bucket_means[i + 1] < 0:
                # 线性插值求零点
                x0, x1 = bucket_centers[i], bucket_centers[i + 1]
                y0, y1 = bucket_means[i], bucket_means[i + 1]
                thresh = x0 - y0 * (x1 - x0) / (y1 - y0)
                return float(thresh)
        return None
    except Exception:  # noqa: BLE001
        return None


def _shap_dependence_grid(shap_values, X, feature_names: list[str]) -> list[str]:
    """为每个特征生成 SHAP Dependence 散点图（含非线性阈值标注），返回 base64 列表。"""
    import io
    import base64

    try:
        import numpy as np

        plt = _configure_mpl()

        results = []
        # 找与每个特征 SHAP 相关性最高的"交互颜色"特征
        mean_abs = np.abs(shap_values).mean(axis=0)
        color_by_idx = int(np.argmax(mean_abs))

        for i, name in enumerate(feature_names):
            fig, ax = plt.subplots(figsize=(6, 4))
            x_vals = X[:, i]
            s_vals = shap_values[:, i]
            color_vals = X[:, color_by_idx]

            sc = ax.scatter(
                x_vals, s_vals, c=color_vals, cmap="RdBu_r", alpha=0.5, s=12
            )
            plt.colorbar(sc, ax=ax, label=feature_names[color_by_idx])
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")

            thresh = _detect_shap_threshold(x_vals, s_vals)
            if thresh is not None:
                ax.axvline(
                    thresh,
                    color="#e74c3c",
                    linewidth=1.5,
                    linestyle="--",
                    label=f"阈值≈{thresh:.3g}",
                )
                ax.legend(fontsize=8)

            ax.set_xlabel(name)
            ax.set_ylabel("SHAP 值")
            ax.set_title(f"Dependence: {name}")
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            plt.close("all")
            buf.seek(0)
            results.append(base64.b64encode(buf.read()).decode())
        return results
    except Exception:  # noqa: BLE001
        return []


def _shap_heatmap_by_time(shap_values, X, feature_names: list[str], buy_dates) -> str:
    """绘制按时间排序的 SHAP 热图（行=样本，列=特征），返回 base64 PNG。"""
    import io
    import base64

    try:
        import numpy as np

        plt = _configure_mpl()

        n_samples = shap_values.shape[0]
        n_feat = len(feature_names)
        # 已按 buy_date 升序排列（_prepare_xy 中已 sort）

        # 为防止数据量太大，最多显示 500 行（均匀采样）
        if n_samples > 500:
            idx = np.linspace(0, n_samples - 1, 500, dtype=int)
            sv = shap_values[idx]
            dates = buy_dates[idx]
        else:
            sv = shap_values
            dates = buy_dates

        fig, ax = plt.subplots(figsize=(max(6, n_feat * 1.2), 6))
        vmax = float(np.abs(sv).max()) or 1.0
        im = ax.imshow(
            sv,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(feature_names, rotation=30, ha="right", fontsize=8)

        # Y 轴显示季度刻度
        if len(dates) > 0:
            try:
                import pandas as pd

                pd_dates = pd.to_datetime(dates.astype(str), errors="coerce")
                quarters = pd_dates.to_period("Q").astype(str)
                uniq_q, q_idx = np.unique(quarters, return_index=True)
                ax.set_yticks(q_idx)
                ax.set_yticklabels([str(q) for q in quarters[q_idx]], fontsize=7)
            except Exception:  # noqa: BLE001
                ax.set_ylabel("样本（时间顺序）")

        plt.colorbar(im, ax=ax, shrink=0.8, label="SHAP 值")
        ax.set_title("SHAP 时间热图（行=交易，列=特征）")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close("all")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:  # noqa: BLE001
        return ""


def _shap_interaction_plots(
    model, X_sample, feature_names: list[str], max_rows: int = 100
) -> str:
    """计算 SHAP Interaction Values 并生成热图+散点双子图，返回 base64 PNG。"""
    import io
    import base64

    try:
        import shap
        import numpy as np

        plt = _configure_mpl()

        X_use = X_sample[:max_rows]
        explainer = shap.TreeExplainer(model)
        inter = explainer.shap_interaction_values(X_use)
        # 分类器返回 list
        if isinstance(inter, list):
            inter = inter[1]

        n_feat = len(feature_names)
        mean_inter = np.abs(inter).mean(axis=0)  # (n_feat, n_feat)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左：interaction 均值热图
        im = axes[0].imshow(mean_inter, cmap="Blues", interpolation="nearest")
        axes[0].set_xticks(range(n_feat))
        axes[0].set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
        axes[0].set_yticks(range(n_feat))
        axes[0].set_yticklabels(feature_names, fontsize=8)
        plt.colorbar(im, ax=axes[0], shrink=0.8, label="mean |interaction|")
        axes[0].set_title("特征交互强度热图")

        # 右：最强交互对散点图
        np.fill_diagonal(mean_inter, 0)
        idx_flat = int(np.argmax(mean_inter))
        fi, fj = divmod(idx_flat, n_feat)
        inter_pair = inter[:, fi, fj]
        axes[1].scatter(
            X_use[:, fi], inter_pair, alpha=0.5, s=12, c=X_use[:, fj], cmap="RdBu_r"
        )
        axes[1].axhline(0, color="#888", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel(feature_names[fi])
        axes[1].set_ylabel(f"SHAP Interaction with {feature_names[fj]}")
        axes[1].set_title(f"最强交互对: {feature_names[fi]} × {feature_names[fj]}")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close("all")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:  # noqa: BLE001
        return ""


def _shap_decision_plots(
    explainer, shap_vals_worst, X_worst, feature_names: list[str]
) -> str:
    """为最差交易绘制 SHAP Decision Plot，返回 base64 PNG。"""
    import io
    import base64

    try:
        import shap

        plt = _configure_mpl()

        expected_value = explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]

        plt.figure(figsize=(8, 6))
        shap.decision_plot(
            expected_value,
            shap_vals_worst,
            X_worst,
            feature_names=feature_names,
            show=False,
        )
        plt.title("最差 N 笔交易 Decision Plot")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        plt.close("all")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:  # noqa: BLE001
        return ""
