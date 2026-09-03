from .base import Teacher
from .gate import GateResult, evaluate
from .shallow import ShallowMLP

__all__ = ["Teacher", "ShallowMLP", "GateResult", "evaluate"]
