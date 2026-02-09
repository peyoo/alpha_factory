# ruff: noqa: E402
# shim: compatibility layer for src/alpha_factory
# Keep this minimal and temporary. Exports the alpha_factory package
from importlib import import_module
# Import the real package so top-level `import alpha` remains valid
_real = import_module('alpha_factory')
# Re-export common names (shallow) to maintain backward compatibility
from alpha_factory import *  # noqa: F401,F403
# Note: remove this shim after all callers have migrated to `alpha_factory`.
