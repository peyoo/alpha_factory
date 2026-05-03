"""tests/test_expr_condition_opt.py — ExprCondition 可优化参数功能单元测试

覆盖：
  - ExprCondition.has_opt_params：有/无 opt 参数时的布尔值
  - ExprCondition.resolved_expression：用 default 值替换占位符
  - ExprCondition.resolve_expression(trial)：用 trial 采样值替换占位符
  - resolved_expression 缺少 default 时抛 ValueError
  - StrategyConfig.has_condition_opt_params：任意条件有 opt 则为 True
  - StrategyConfig.get_static_condition_exprs：只收集无 opt 参数的条件
  - StrategyConfig.apply_best_condition_params：更新 opt.default
  - StrategyConfig.get_condition_exprs：使用 resolved_expression
  - StrategyConfig.opt_mode：默认 False，不序列化到 model_dump
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alpha_factory.config.strategy import ExprCondition, StrategyConfig


# ---------------------------------------------------------------------------
# ExprCondition — has_opt_params
# ---------------------------------------------------------------------------


def test_has_opt_params_false_when_no_opt():
    cond = ExprCondition(expression="TOTAL_MV > 1e9", name="c1")
    assert cond.has_opt_params is False


def test_has_opt_params_true_when_opt_present():
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="c1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9}},
    )
    assert cond.has_opt_params is True


# ---------------------------------------------------------------------------
# ExprCondition — resolved_expression
# ---------------------------------------------------------------------------


def test_resolved_expression_no_opt():
    cond = ExprCondition(expression="TOTAL_MV > 1e9", name="c1")
    assert cond.resolved_expression == "TOTAL_MV > 1e9"


def test_resolved_expression_with_default():
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="c1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1234567890.0}},
    )
    assert cond.resolved_expression == "TOTAL_MV > 1234567890.0"


def test_resolved_expression_multiple_params():
    cond = ExprCondition(
        expression="LIST_DAYS > {min_days} and TOTAL_MV > {v1}",
        name="c2",
        opt={
            "min_days": {"type": "int", "low": 60, "high": 500, "default": 120},
            "v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9},
        },
    )
    result = cond.resolved_expression
    assert "{min_days}" not in result
    assert "{v1}" not in result
    assert "120" in result
    assert "1000000000.0" in result


def test_resolved_expression_raises_when_no_default():
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="c1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9}},  # 故意缺少 default
    )
    with pytest.raises(ValueError, match="default"):
        _ = cond.resolved_expression


# ---------------------------------------------------------------------------
# ExprCondition — resolve_expression(trial)
# ---------------------------------------------------------------------------


def _mock_trial(suggest_values: dict) -> MagicMock:
    """构造返回固定值的 mock trial。"""
    trial = MagicMock()
    trial.suggest_float.side_effect = lambda name, *a, **kw: suggest_values.get(
        name, 0.0
    )
    trial.suggest_int.side_effect = lambda name, *a, **kw: suggest_values.get(name, 0)
    trial.suggest_categorical.side_effect = lambda name, choices: suggest_values.get(
        name, choices[0]
    )
    return trial


def test_resolve_expression_float():
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="mask1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9}},
    )
    trial = _mock_trial({"mask1__v1": 2000000000.0})
    result = cond.resolve_expression(trial)
    assert result == "TOTAL_MV > 2000000000.0"
    trial.suggest_float.assert_called_once_with("mask1__v1", 5e8, 3e9, step=None)


def test_resolve_expression_int():
    cond = ExprCondition(
        expression="LIST_DAYS > {min_days}",
        name="mask2",
        opt={
            "min_days": {
                "type": "int",
                "low": 60,
                "high": 500,
                "step": 10,
                "default": 120,
            }
        },
    )
    trial = _mock_trial({"mask2__min_days": 250})
    result = cond.resolve_expression(trial)
    assert result == "LIST_DAYS > 250"
    trial.suggest_int.assert_called_once_with("mask2__min_days", 60, 500, step=10)


def test_resolve_expression_categorical():
    cond = ExprCondition(
        expression="MARKET_TYPE == {mtype}",
        name="mask3",
        opt={
            "mtype": {"type": "categorical", "choices": ["A", "B", "C"], "default": "A"}
        },
    )
    trial = _mock_trial({"mask3__mtype": "B"})
    result = cond.resolve_expression(trial)
    assert result == "MARKET_TYPE == B"


def test_resolve_expression_invalid_type():
    cond = ExprCondition(
        expression="X > {v}",
        name="c",
        opt={"v": {"type": "unknown", "default": 1}},
    )
    with pytest.raises(ValueError, match="type="):
        cond.resolve_expression(MagicMock())


def test_resolve_expression_no_opt_returns_original():
    cond = ExprCondition(expression="TOTAL_MV > 1e9", name="c")
    result = cond.resolve_expression(MagicMock())
    assert result == "TOTAL_MV > 1e9"


# ---------------------------------------------------------------------------
# StrategyConfig — opt_mode 字段
# ---------------------------------------------------------------------------


def test_opt_mode_default_false():
    cfg = StrategyConfig()
    assert cfg.opt_mode is False


def test_opt_mode_not_serialized():
    cfg = StrategyConfig()
    cfg.opt_mode = True
    dumped = cfg.model_dump()
    assert "opt_mode" not in dumped


# ---------------------------------------------------------------------------
# StrategyConfig — has_condition_opt_params
# ---------------------------------------------------------------------------


def _cfg_with_opt_condition() -> StrategyConfig:
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="pool_mask1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9}},
    )
    return StrategyConfig(pool_mask=[cond])


def test_has_condition_opt_params_true():
    cfg = _cfg_with_opt_condition()
    assert cfg.has_condition_opt_params() is True


def test_has_condition_opt_params_false_when_no_conditions():
    cfg = StrategyConfig()
    assert cfg.has_condition_opt_params() is False


def test_has_condition_opt_params_false_when_no_opt():
    cond = ExprCondition(expression="TOTAL_MV > 1e9", name="pool_mask1")
    cfg = StrategyConfig(pool_mask=[cond])
    assert cfg.has_condition_opt_params() is False


# ---------------------------------------------------------------------------
# StrategyConfig — get_static_condition_exprs
# ---------------------------------------------------------------------------


def test_get_static_condition_exprs_excludes_opt_conditions():
    cond_static = ExprCondition(expression="LIST_DAYS > 100", name="pool_mask1")
    cond_dynamic = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="pool_mask2",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9}},
    )
    cfg = StrategyConfig(pool_mask=[cond_static, cond_dynamic])
    static_exprs = cfg.get_static_condition_exprs()
    assert len(static_exprs) == 1
    assert static_exprs[0] == "pool_mask1 = LIST_DAYS > 100"


def test_get_static_condition_exprs_empty_when_all_dynamic():
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="pool_mask1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9}},
    )
    cfg = StrategyConfig(pool_mask=[cond])
    assert cfg.get_static_condition_exprs() == []


# ---------------------------------------------------------------------------
# StrategyConfig — get_condition_exprs（使用 resolved_expression）
# ---------------------------------------------------------------------------


def test_get_condition_exprs_uses_default_value():
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="pool_mask1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 999.0}},
    )
    cfg = StrategyConfig(pool_mask=[cond])
    exprs = cfg.get_condition_exprs()
    assert len(exprs) == 1
    assert "999.0" in exprs[0]
    assert "{v1}" not in exprs[0]


def test_get_condition_exprs_skips_empty_expression():
    cond = ExprCondition(expression="", name="pool_mask1")
    cfg = StrategyConfig(pool_mask=[cond])
    assert cfg.get_condition_exprs() == []


# ---------------------------------------------------------------------------
# StrategyConfig — apply_best_condition_params
# ---------------------------------------------------------------------------


def test_apply_best_condition_params_updates_default():
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="pool_mask1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9}},
    )
    cfg = StrategyConfig(pool_mask=[cond])
    cfg.apply_best_condition_params({"pool_mask1__v1": 2.5e9})
    assert cfg.pool_mask[0].opt["v1"]["default"] == 2.5e9


def test_apply_best_condition_params_ignores_missing_keys():
    cond = ExprCondition(
        expression="TOTAL_MV > {v1}",
        name="pool_mask1",
        opt={"v1": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9}},
    )
    cfg = StrategyConfig(pool_mask=[cond])
    # 提供不相关的 key，original default 不变
    cfg.apply_best_condition_params({"other_cond__v2": 999.0})
    assert cfg.pool_mask[0].opt["v1"]["default"] == 1e9


def test_apply_best_condition_params_multiple_groups():
    buy_cond = ExprCondition(
        expression="LIST_DAYS > {d}",
        name="buy_able1",
        opt={"d": {"type": "int", "low": 60, "high": 500, "default": 100}},
    )
    not_buy_cond = ExprCondition(
        expression="TOTAL_MV > {v}",
        name="not_buy_able1",
        opt={"v": {"type": "float", "low": 5e8, "high": 3e9, "default": 1e9}},
    )
    cfg = StrategyConfig(buy_able=[buy_cond], not_buy_able=[not_buy_cond])
    cfg.apply_best_condition_params(
        {
            "buy_able1__d": 200,
            "not_buy_able1__v": 1.5e9,
        }
    )
    assert cfg.buy_able[0].opt["d"]["default"] == 200
    assert cfg.not_buy_able[0].opt["v"]["default"] == 1.5e9
