from ._version import __version__
from .api import SolveResult, SolveSummary, solve, solve_aim
from .core import BoundaryConditionSet, Model, OutputProfile, SolverProfile
from .load_cases import AxialCompression, BodyWeightCompression, SimpleShear

__all__ = [
    "AxialCompression",
    "BoundaryConditionSet",
    "BodyWeightCompression",
    "Model",
    "OutputProfile",
    "SimpleShear",
    "SolveResult",
    "SolveSummary",
    "SolverProfile",
    "__version__",
    "solve",
    "solve_aim",
]
