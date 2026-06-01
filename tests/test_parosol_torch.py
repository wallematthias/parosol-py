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
from parosol_torch.registry import available_backends, get_backend, solve


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


def test_experimental_backend_registry_is_non_default():
    names = available_backends()

    assert names == ("torch-experimental",)
    assert "native" not in names
    assert "parosol_native" not in names

    backend = get_backend("torch-experimental")

    assert backend.status is BackendStatus.EXPERIMENTAL


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


def test_torch_experimental_backend_solves_single_element_on_cpu():
    pytest.importorskip("torch")
    result = solve(
        _single_element_tension_problem(),
        SolverSettings(tolerance=1e-8, max_iterations=100, device="cpu"),
    )

    assert result.converged
    assert result.iterations is not None
    assert result.residual_norm is not None
    assert result.fields["displacements"].shape == (2, 2, 2, 3)
    assert result.fields["forces"].shape == (2, 2, 2, 3)
    assert result.fields["sed"].shape == (1, 1, 1)
    assert np.allclose(result.fields["displacements"][0, :, :, :], 0.0)
    assert np.allclose(result.fields["displacements"][1, :, :, 0], 0.01)
    assert float(result.fields["sed"][0, 0, 0]) > 0
    assert result.diagnostics["device"] == "cpu"


def test_torch_experimental_backend_solves_single_element_on_mps_when_available():
    pytest.importorskip("torch")
    if not is_available("mps"):
        pytest.skip("MPS is not available")

    result = solve(
        _single_element_tension_problem(),
        SolverSettings(tolerance=1e-5, max_iterations=100, device="mps"),
    )

    assert result.converged
    assert result.diagnostics["device"] == "mps"
    assert np.allclose(result.fields["displacements"][1, :, :, 0], 0.01, atol=1e-5)


def test_torch_experimental_backend_solves_single_element_on_cuda_when_available():
    pytest.importorskip("torch")
    if not is_available("cuda"):
        pytest.skip("CUDA is not available")

    result = solve(
        _single_element_tension_problem(),
        SolverSettings(tolerance=1e-5, max_iterations=100, device="cuda"),
    )

    assert result.converged
    assert result.diagnostics["device"] == "cuda"
    assert np.allclose(result.fields["displacements"][1, :, :, 0], 0.01, atol=1e-5)


def _single_element_tension_problem() -> VoxelElasticityProblem:
    fixed = []
    loaded = []
    for y in range(2):
        for z in range(2):
            for component in range(3):
                fixed.append([0, y, z, component])
            loaded.append([1, y, z, 0])
    return VoxelElasticityProblem(
        stiffness_gpa_xyz=np.ones((1, 1, 1), dtype=np.float32),
        voxel_size_mm=1.0,
        poisson_ratio=0.3,
        fixed_displacement_coordinates=np.asarray(fixed, dtype=np.int64),
        fixed_displacement_values=np.zeros(len(fixed), dtype=np.float64),
        loaded_node_coordinates=np.asarray(loaded, dtype=np.int64),
        loaded_node_values=np.full(len(loaded), 0.01, dtype=np.float64),
        requested_outputs=("forces", "displacements", "sed"),
    )
