"""Experimental torch-backed solver namespace for ParOSol-py.

This package is intentionally separate from :mod:`parosol_py` and the bundled
native ParOSol executable. It will host accelerator backends only after they
can be validated against the native/reference solver.
"""

from .backend import TorchBackendInfo, backend_info, is_available
from .contract import (
    BackendStatus,
    SolverSettings,
    VoxelElasticityProblem,
    VoxelElasticityResult,
)
from .registry import available_backends, get_backend, solve

__all__ = [
    "BackendStatus",
    "SolverSettings",
    "TorchBackendInfo",
    "VoxelElasticityProblem",
    "VoxelElasticityResult",
    "available_backends",
    "backend_info",
    "get_backend",
    "is_available",
    "solve",
]
