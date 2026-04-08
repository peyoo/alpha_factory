# tests/test_cli_loader.py
import pytest
from alpha_factory.cli._loader import resolve_pool


def test_resolve_pool_known():
    pool = resolve_pool("main_small_pool")
    assert pool.name == "main_small_pool"


def test_resolve_pool_unknown_raises():
    with pytest.raises(ValueError, match="未知股票池"):
        resolve_pool("nonexistent_pool")


def test_resolve_pool_error_message_lists_valid_options():
    with pytest.raises(ValueError) as exc_info:
        resolve_pool("bad_pool")
    assert "main_small_pool" in str(exc_info.value)
