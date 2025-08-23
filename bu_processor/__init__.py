"""Top-level bu_processor package shim.

The repository layout contains a nested directory structure:

	bu_processor/            (outer package, tests live alongside)
		__init__.py          (this file)
		bu_processor/        (inner actual implementation package)
			pipeline/
			training/
			core/
			...

To allow imports like `from bu_processor.pipeline import ...` to work, we
append the inner implementation directory to this package's __path__ so that
Python can resolve submodules under the inner folder transparently. This
avoids a disruptive repository restructure while restoring expected import
paths for the existing tests.
"""

from pathlib import Path as _Path
import sys as _sys

_outer_dir = _Path(__file__).parent
_inner_dir = _outer_dir / "bu_processor"

# Dynamically extend package search path if inner implementation folder exists
if _inner_dir.exists() and _inner_dir.is_dir():  # pragma: no cover (import-time)
	# Ensure the inner directory is importable as a namespace continuation
	if str(_inner_dir) not in __path__:  # type: ignore # hasattr provided by import system
		__path__.append(str(_inner_dir))  # type: ignore

	# Optional: expose commonly used symbols from inner package root if present
	try:  # pragma: no cover - defensive
		from .bu_processor import __all__ as _inner_all  # type: ignore
		from .bu_processor import *  # noqa: F401,F403
		__all__ = list(set(_inner_all))  # type: ignore
	except Exception:  # pragma: no cover
		# Fallback: leave __all__ undefined if inner import fails
		pass

# Explicitly import key classes for easy access
try:
	from .bu_processor import EnhancedIntegratedPipeline
except ImportError:
	pass
