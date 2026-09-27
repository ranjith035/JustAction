"""
Jev System 1 Decision Engine
----------------------------
A deterministic, non-autoregressive decision model built from scratch.
Implements the core primitives of Jev (Choice, Score, Noul),
RLCD-inspired calibration, and strictly typed zero-prose action emission.
"""

from jev.core.model import JevModel
from jev.core.config import JevConfig
from jev.runtime.engine import JevActionEngine
from jev.schema.types import ActionOutput, DecisionOutput, NoulDecision

__all__ = [
    "JevModel",
    "JevConfig",
    "JevActionEngine",
    "ActionOutput",
    "DecisionOutput",
    "NoulDecision",
]

__version__ = "0.1.0"
