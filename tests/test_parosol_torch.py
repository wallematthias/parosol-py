import numpy as np
import pytest

import parosol_py
import parosol_torch
from parosol_torch import backend_info, is_available
from parosol_torch.backend import TorchBackendInfo
from parosol_torch.contract import (
    BackendStatus,
    SolverSettings,
    VoxelElasticityProblem,
)
from parosol_torch.prototype import apply_scalar_poisson_7point
from parosol_torch.registry import available_backends, get_backend


def test_parosol_torch_namespace_is_separate_from_native_solver():
    info = backend_info()

    assert isinstance(info, TorchBackendInfo)
    assert isinstance(info.torch_installed, bool)
    assert isinstance(info.mps_available, bool)
    assert isinstance(info.cuda_available, bool)
    assert "parosol_native" not in info.status
    assert parosol_torch.solve is not parosol_py.solve


def test_parosol_torch_device_availability_is_explicit():
    assert isinstance(is_available(), bool)
    assert isinstance(is_available("mps"), bool)
    assert is_available("unknown-device") is False


def test_backend_info_handles_missing_torch(monkeypatch):
    import parosol_torch.backend as backend

    def fake_find_spec(name):
        if name == "torch":
            return None
        return object()

    monkeypatch.setattr(backend.util, "find_spec", fake_find_spec)

    info = backend.backend_info()

    assert info.torch_installed is False
    assert info.available is False
    assert info.default_device is None
    assert is_available("cpu") is False


def test_voxel_elasticity_contract_can_be_constructed():
    problem = VoxelElasticityProblem(
        stiffness_gpa_xyz=np.ones((2, 2, 2), dtype=np.float32),
        voxel_size_mm=0.061,
        poisson_ratio=0.3,
        fixed_displacement_coordinates=np.array([[0, 0, 0, 0]], dtype=np.int64),
        fixed_displacement_values=np.array([0.0], dtype=np.float64),
        requested_outputs=("forces", "displacements"),
    )
    settings = SolverSettings(tolerance=1e-5, max_iterations=25, device="cpu")

    assert problem.dimensions_xyz == (2, 2, 2)
    assert settings.tolerance == pytest.approx(1e-5)
    assert settings.device == "cpu"


def test_experimental_backend_registry_is_non_default_and_fails_clearly():
    names = available_backends()

    assert names == ("torch-experimental",)
    assert "native" not in names
    assert "parosol_native" not in names

    backend = get_backend("torch-experimental")

    assert backend.status is BackendStatus.NOT_IMPLEMENTED
    with pytest.raises(NotImplementedError, match="not a validated ParOSol solver"):
        backend.solve(
            VoxelElasticityProblem(
                stiffness_gpa_xyz=np.ones((1, 1, 1), dtype=np.float32),
                voxel_size_mm=1.0,
                poisson_ratio=0.3,
                fixed_displacement_coordinates=np.array([[0, 0, 0, 0]]),
                fixed_displacement_values=np.array([0.0]),
            ),
            SolverSettings(),
        )


def test_unknown_backend_fails_without_fallback():
    with pytest.raises(KeyError, match="unknown parosol_torch backend"):
        get_backend("native")


def test_scalar_poisson_prototype_applies_7point_operator():
    values = np.zeros((3, 3, 3), dtype=np.float64)
    values[1, 1, 1] = 2.0

    out = apply_scalar_poisson_7point(values, spacing=0.5)

    assert out[1, 1, 1] == pytest.approx(48.0)
    assert out[0, 1, 1] == pytest.approx(-8.0)
    assert out[1, 0, 1] == pytest.approx(-8.0)
    assert out[1, 1, 0] == pytest.approx(-8.0)
