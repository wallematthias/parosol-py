from ._version import __version__
from .api import SolveResult, SolveSummary, solve, solve_aim
from .core import BoundaryConditionSet, Model, OutputProfile, SolverProfile

__all__ = [
    "BoundaryConditionSet",
    "Model",
    "OutputProfile",
    "SolveResult",
    "SolveSummary",
    "SolverProfile",
    "__version__",
    "solve",
    "solve_aim",
]
