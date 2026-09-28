"""Neural Architecture Search (NAS) Package.

This package provides foundational NAS engine abstractions, optimization loops,
and callback mechanisms built on top of Pymoo for multi-objective optimization.
"""

from .base_nas import BaseNAS, NASHistoryCallback

__all__ = ["BaseNAS", "NASHistoryCallback"]