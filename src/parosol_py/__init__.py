from ._version import __version__
from .api import SolveResult, SolveSummary, solve, solve_aim
from .core import BoundaryConditionSet, Model, OutputProfile, SolverProfile
from .load_cases import AxialCompression, BodyWeightCompression, SimpleShear
from .profiles import get_output_profile, get_solver_profile

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
    "get_output_profile",
    "get_solver_profile",
    "solve",
    "solve_aim",
]
