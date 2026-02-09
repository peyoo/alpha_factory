import importlib
import sys


def test_alpha_factory_package_importable():
    """
    Smoke test:
    - ensure top-level package `alpha_factory` can be imported
    - ensure a representative subpackage `alpha_factory.data_provider` imports
    This validates the src/ layout + editable install or Poetry environment.
    """
    # Attempt import of the main package
    mod = importlib.import_module("alpha_factory")
    assert mod is not None
    # Basic sanity check: subpackage exists and can be imported
    dp = importlib.import_module("alpha_factory.data_provider")
    assert dp is not None

    # Optional: check a common attribute or module name (non-strict)
    assert "alpha_factory" in sys.modules or mod.__name__ == "alpha_factory"

