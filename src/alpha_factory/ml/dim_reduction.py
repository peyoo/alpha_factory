"""
因子筛选分析模块
================
基于面板数据的正则化回归因子筛选：

使用 DataProvider 加载面板数据（DATE × ASSET），以因子值（f1~fN）为特征列，
以前瞻收益（LABEL_FOR_IC）为预测目标，通过 SGDRegressor 正则化回归
筛选出对收益有贡献的因子，剔除系数为 0 的冗余因子。
"""

from __future__ import annotations

import base64
import io
import webbrowser
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import polars as pl
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score as _r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler

# ──────────────────────────────────────────────
# 元数据列：evals CSV 中非数值 / 非指标列
# ──────────────────────────────────────────────
_META_COLS = {"factor", "expression", "direction"}

# evals CSV 中常见数值指标列（按优先级排列）
_DEFAULT_METRIC_COLS = [
    "ic_mean",
    "ic_mean_abs",
    "ic_ir",
    "ic_ir_abs",
    "ann_ret",
    "sharpe",
    "turnover_est",
]


# ──────────────────────────────────────────────
# 返回结构
# ──────────────────────────────────────────────
@dataclass
class FactorSelectionResult:
    """正则化因子筛选结果（Lasso / Elastic Net 通用）。"""

    method: str  # "lasso" 或 "elastic-net"
    factor_names: list[str]  # 所有参与回归的因子名
    expressions: dict[str, str]  # 因子名 -> 表达式 映射
    # ── 回归结果 ─────────────────────────────────
    factor_coefs: dict[str, float]  # 因子名 -> 回归系数
    selected_factors: list[str]  # 系数非零的因子（按 |coef| 降序）
    eliminated_factors: list[str]  # 系数为零被剔除的因子
    alpha: float  # CV 最优正则化强度
    l1_ratio: float  # Elastic Net L1 比例（Lasso 时为 1.0）
    r2: float  # 拟合优度 R²
    n_samples: int  # 参与回归的样本数
    y_true: np.ndarray  # 目标真实值
    y_pred: np.ndarray  # 模型预测值
    target_col: str  # 目标列名


# ──────────────────────────────────────────────
# 数据加载（保留：evals CSV 工具函数）
# ──────────────────────────────────────────────


def load_factor_csv(path: Path) -> pl.DataFrame:
    """加载因子 CSV 文件，返回 Polars DataFrame。"""
    return pl.read_csv(path)


def detect_metric_cols(df: pl.DataFrame) -> list[str]:
    """
    自动推断数值型指标列。
    优先返回已知指标列（与 CSV 中同名的），其余数值列按序追加。
    """
    numeric_types = {
        pl.Float32,
        pl.Float64,
        pl.Int16,
        pl.Int32,
        pl.Int64,
    }
    all_numeric = [
        c
        for c, t in zip(df.columns, df.dtypes)
        if t in numeric_types and c not in _META_COLS
    ]
    # 已知指标列优先（保持预期顺序）
    known = [c for c in _DEFAULT_METRIC_COLS if c in all_numeric]
    extra = [c for c in all_numeric if c not in known]
    return known + extra


# ──────────────────────────────────────────────
# 矩阵构建
# ──────────────────────────────────────────────


def build_metrics_matrix(
    df: pl.DataFrame,
    metric_cols: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    metrics 模式：每行=因子，每列=性能指标。
    返回 (X, factor_names, feature_names)，X 已去除含 NaN 的行。
    """
    factor_col = "factor" if "factor" in df.columns else df.columns[0]
    sub = df.select([factor_col] + metric_cols).drop_nulls()
    factor_names = sub[factor_col].to_list()
    X = sub.select(metric_cols).to_numpy().astype(np.float64)
    return X, factor_names, metric_cols


# ──────────────────────────────────────────────
# 正则化面板回归因子筛选（核心）
# ──────────────────────────────────────────────


def run_regularized_selection(
    panel_df: pl.DataFrame,
    factor_cols: list[str],
    target_col: str = "LABEL_FOR_IC",
    method: Literal["lasso", "elastic-net"] = "elastic-net",
    l1_ratio: float = 0.5,
    cv: int = 5,
    pool_mask_col: Optional[str] = "POOL_MASK",
    max_samples: int = 200_000,
) -> FactorSelectionResult:
    """
    面板数据正则化回归因子筛选。

    以 ``factor_cols``（如 f1~f165）为特征列，以 ``target_col``（如 LABEL_FOR_IC，
    前瞻收益）为预测目标，通过 SGDRegressor（lasso/elastic-net）回归，输出每个因子的系数。
    系数为 0 的因子被剔除，保留对收益有贡献的因子。

    Parameters
    ----------
    panel_df        : 面板 DataFrame（DATE x ASSET x factor_values + target + pool_mask）
    factor_cols     : 特征列名列表（因子名，如 ["f1", "f2", ...]）
    target_col      : 目标列名（默认 "LABEL_FOR_IC"）
    method          : "lasso" 或 "elastic-net"
    l1_ratio        : Elastic Net L1 比例（仅 elastic-net 生效；0=Ridge, 1=Lasso）
    cv              : 交叉验证折数
    pool_mask_col   : 股票池掩码列（None 则不过滤）
    max_samples     : 最大样本行数（超过则随机抽样），0 表示不限制

    Returns
    -------
    FactorSelectionResult
    """
    # ── 输入校验 ──
    if target_col not in panel_df.columns:
        raise ValueError(
            f"target_col={target_col!r} 不在 DataFrame 列中。"
            f"可用列：{panel_df.columns[:20]}..."
        )
    missing = [c for c in factor_cols if c not in panel_df.columns]
    if missing:
        raise ValueError(f"以下因子列在 DataFrame 中不存在：{missing[:10]}...")

    # ── 过滤股票池 & 去 NaN ──
    df = panel_df
    if pool_mask_col and pool_mask_col in df.columns:
        df = df.filter(pl.col(pool_mask_col))

    select_cols = factor_cols + [target_col]
    df = df.select(select_cols).drop_nulls()

    # 将 ±inf 替换为 null 后再次 drop，避免 sklearn 报错
    float_cols = [c for c in select_cols if df.schema[c] in (pl.Float32, pl.Float64)]
    if float_cols:
        df = df.with_columns(
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in float_cols
        ).drop_nulls()

    n_valid = len(df)
    assert n_valid >= 10, f"有效样本过少（{n_valid} 行），无法进行回归"

    # ── 大数据集随机抽样 ──
    if max_samples > 0 and n_valid > max_samples:
        df = df.sample(n=max_samples, seed=42)
        n_valid = max_samples

    # ── 构建 numpy 矩阵 ──
    X = df.select(factor_cols).to_numpy().astype(np.float64)
    y = df[target_col].to_numpy().astype(np.float64)

    # ── 极值处理（Winsorize）：X 每列 + y 均截断至 1%~99%，防止梯度爆炸 ──
    lo, hi = np.percentile(X, [1, 99], axis=0)  # shape (n_features,)
    X = np.clip(X, lo, hi)
    y_lo, y_hi = np.percentile(y, [1, 99])
    y = np.clip(y, y_lo, y_hi)

    # ── 缩放（RobustScaler 对残余离群值更鲁棒） ──
    scaler_X = RobustScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # ── 回归 ──
    cv_folds = min(cv, n_valid - 1)
    actual_l1_ratio: float
    common_params = {
        "max_iter": 5000,
        "tol": 1e-3,
        "random_state": 42,
        "learning_rate": "adaptive",  # 自动调速，防止梯度飞掉
        "eta0": 0.001,  # 初始步长保守设定，避免早期发散
        "early_stopping": True,  # 大数据必开，防止无谓迭代
        "validation_fraction": 0.1,
    }
    if method == "lasso":
        actual_l1_ratio = 1.0
        base_model = SGDRegressor(penalty="l1", **common_params)
    elif method == "elastic-net":
        actual_l1_ratio = l1_ratio if l1_ratio > 0 else 0.5
        base_model = SGDRegressor(
            penalty="elasticnet",
            l1_ratio=actual_l1_ratio,
            **common_params,
        )
    else:
        raise ValueError(f"method={method!r} 不支持，可选：lasso / elastic-net")

    model = GridSearchCV(
        base_model,
        param_grid={"alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
        cv=min(3, cv_folds),
        scoring="r2",
        n_jobs=-1,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message="Objective did not converge")
        model.fit(X_s, y_s)

    # ── 拆出最优子模型（SGD elastic-net 通过 GridSearchCV 包装） ──
    _best_alpha: float
    if hasattr(model, "best_estimator_"):
        _best_alpha = float(model.best_params_["alpha"])
        model = model.best_estimator_
    else:
        _best_alpha = float(model.alpha_)

    # ── 反标准化预测值 ──
    y_pred_s = model.predict(X_s)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    r2 = float(_r2_score(y, y_pred))

    # ── 构建因子系数映射 ──
    coefs = model.coef_  # standardized 空间系数
    factor_coefs = {name: float(c) for name, c in zip(factor_cols, coefs)}

    # 按 |coef| 降序排列
    selected = [
        name
        for name, c in sorted(
            factor_coefs.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        if abs(c) > 1e-10
    ]
    eliminated = [name for name in factor_cols if name not in set(selected)]

    return FactorSelectionResult(
        method=method,
        factor_names=list(factor_cols),
        expressions={},  # 由 CLI 层注入
        factor_coefs=factor_coefs,
        selected_factors=selected,
        eliminated_factors=eliminated,
        alpha=_best_alpha,
        l1_ratio=actual_l1_ratio,
        r2=r2,
        n_samples=n_valid,
        y_true=y,
        y_pred=y_pred,
        target_col=target_col,
    )


# ──────────────────────────────────────────────
# HTML 报告
# ──────────────────────────────────────────────


def generate_selection_html_report(
    result: FactorSelectionResult,
    output_path: Path,
    open_browser: bool = True,
) -> Path:
    """生成因子筛选 HTML 报告（Lasso / Elastic Net 通用）。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for HTML report generation.") from exc

    method_label = "Lasso" if result.method == "lasso" else "Elastic Net"
    imgs: list[tuple[str, str]] = []

    # ── 图1：因子系数条形图（仅非零） ──
    nonzero = {k: v for k, v in result.factor_coefs.items() if abs(v) > 1e-10}
    if nonzero:
        sorted_items = sorted(
            nonzero.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        names = [k for k, _ in sorted_items]
        coefs_vals = [v for _, v in sorted_items]
        fig1, ax1 = plt.subplots(
            figsize=(10, max(3, len(names) * 0.35)),
        )
        colors = ["#4C72B0" if c >= 0 else "#DD8452" for c in coefs_vals]
        ax1.barh(
            names[::-1],
            coefs_vals[::-1],
            color=colors[::-1],
            alpha=0.85,
        )
        ax1.axvline(0, color="gray", linewidth=0.8)
        ax1.set_xlabel(f"{method_label} Coefficient (standardized)")
        ax1.set_title(
            f"Factor Coefficients ({method_label}, "
            f"{len(nonzero)} selected / {len(result.factor_names)} total, "
            f"R\u00b2={result.r2:.4f})"
        )
        fig1.tight_layout()
        imgs.append(("因子系数分布", _fig_to_b64(fig1)))
        plt.close(fig1)

    # ── 图2：真实 vs 预测散点图（抽样） ──
    n_pts = len(result.y_true)
    max_pts = min(5000, n_pts)
    rng = np.random.default_rng(42)
    if n_pts > max_pts:
        idx = rng.choice(n_pts, size=max_pts, replace=False)
    else:
        idx = np.arange(n_pts)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(
        result.y_true[idx],
        result.y_pred[idx],
        alpha=0.15,
        s=4,
        color="#4C72B0",
    )
    lo = min(result.y_true[idx].min(), result.y_pred[idx].min())
    hi = max(result.y_true[idx].max(), result.y_pred[idx].max())
    ax2.plot([lo, hi], [lo, hi], "r--", linewidth=0.8, label="y=x")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title(f"Actual vs Predicted ({result.target_col}, R\u00b2={result.r2:.4f})")
    ax2.legend()
    fig2.tight_layout()
    imgs.append(("真实 vs 预测", _fig_to_b64(fig2)))
    plt.close(fig2)

    # ── HTML 拼装 ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_sel = len(result.selected_factors)
    n_total = len(result.factor_names)

    # 因子系数表格行
    coef_rows = []
    for name in result.selected_factors:
        coef = result.factor_coefs[name]
        expr = result.expressions.get(name, "")
        coef_rows.append(
            f"<tr><td><b>{name}</b></td><td>{coef:+.6f}</td>"
            f"<td style='font-size:0.8em;color:#636e72'>{expr}</td></tr>"
        )
    coef_table_html = "\n".join(coef_rows)

    # 被剔除因子
    elim_html = (
        ", ".join(result.eliminated_factors) if result.eliminated_factors else "无"
    )

    img_blocks = "\n".join(
        f'<div class="chart-block"><h3>{t}</h3>'
        f'<img src="data:image/png;base64,{b}" alt="{t}" /></div>'
        for t, b in imgs
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <title>{method_label} 因子筛选报告</title>
  <style>
    body {{{{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #f5f6fa; color: #2d3436; margin: 0; padding: 20px; }}}}
    .header {{{{ background: #2d3436; color: #fff; padding: 20px 30px;
               border-radius: 8px; margin-bottom: 20px; }}}}
    .header h1 {{{{ margin: 0; font-size: 1.4em; }}}}
    .header p  {{{{ margin: 4px 0 0; opacity: 0.75; font-size: 0.9em; }}}}
    .meta-table {{{{ border-collapse: collapse; margin: 10px 0 20px; }}}}
    .meta-table td {{{{ padding: 4px 14px; border: 1px solid #ccc; font-size: 0.9em; }}}}
    .chart-block {{{{ background: #fff; border-radius: 8px; padding: 20px;
                    margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}}}
    .chart-block h3 {{{{ margin-top: 0; color: #636e72; }}}}
    .chart-block img {{{{ max-width: 100%; }}}}
    .coef-table {{{{ border-collapse: collapse; width: 100%; margin-top: 10px; }}}}
    .coef-table td, .coef-table th {{{{
        padding: 5px 12px; border: 1px solid #dfe6e9; font-size: 0.85em; }}}}
    .coef-table th {{{{ background: #dfe6e9; }}}}
  </style>
</head>
<body>
  <div class="header">
    <h1>Alpha Factory - {method_label} 因子筛选报告</h1>
    <p>生成时间：{timestamp} | 目标：{result.target_col}
       | 方法：{method_label}
       | 样本数：{result.n_samples:,}</p>
  </div>
  <table class="meta-table">
    <tr><td>筛选方法</td><td>{method_label}</td></tr>
    <tr><td>目标列</td><td>{result.target_col}</td></tr>
    <tr><td>R2</td><td>{result.r2:.6f}</td></tr>
    <tr><td>CV alpha</td><td>{result.alpha:.6f}</td></tr>
    <tr><td>L1 Ratio</td><td>{result.l1_ratio:.4f}</td></tr>
    <tr><td>保留因子数</td><td>{n_sel} / {n_total}</td></tr>
    <tr><td>样本数</td><td>{result.n_samples:,}</td></tr>
  </table>
  {img_blocks}
  <div class="chart-block">
    <h3>保留因子及系数（{n_sel} 个）</h3>
    <table class="coef-table">
      <tr><th>因子</th><th>系数</th><th>表达式</th></tr>
      {coef_table_html}
    </table>
  </div>
  <div class="chart-block">
    <h3>被剔除因子（{len(result.eliminated_factors)} 个）</h3>
    <p style="font-size:0.85em;color:#636e72">{elim_html}</p>
  </div>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())
    return output_path


def _fig_to_b64(fig) -> str:
    """将 matplotlib Figure 转成 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
