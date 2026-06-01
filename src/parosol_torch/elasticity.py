from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .backend import require_backend
from .contract import (
    BackendStatus,
    SolverSettings,
    VoxelElasticityProblem,
    VoxelElasticityResult,
)


@dataclass(frozen=True)
class TorchVoxelElasticityBackend:
    """Experimental matrix-free torch backend for voxel linear elasticity."""

    name: str = "torch-experimental"
    status: BackendStatus = BackendStatus.EXPERIMENTAL

    def solve(
        self,
        problem: VoxelElasticityProblem,
        settings: SolverSettings | None = None,
    ) -> VoxelElasticityResult:
        settings = settings or SolverSettings()
        return solve_voxel_elasticity(problem, settings)


def solve_voxel_elasticity(
    problem: VoxelElasticityProblem,
    settings: SolverSettings | None = None,
) -> VoxelElasticityResult:
    """Solve a small voxel elasticity problem with torch CG.

    This backend is intentionally explicit and experimental. It implements the
    standard trilinear 8-node hexahedral stiffness action with homogeneous or
    prescribed displacement boundary conditions. It is suitable for backend
    development and small validation cases, not yet for production HR-pQCT.
    """

    settings = settings or SolverSettings()
    torch = require_backend(settings.device or "cpu")
    device = _select_device(torch, settings.device)
    dtype = torch.float64 if str(device) == "cpu" else torch.float32
    started = perf_counter()

    stiffness = torch.as_tensor(
        np.asarray(problem.stiffness_gpa_xyz, dtype=np.float32),
        dtype=dtype,
        device=device,
    )
    if stiffness.ndim != 3:
        raise ValueError("stiffness_gpa_xyz must be a 3D xyz array")
    if float(problem.voxel_size_mm) <= 0:
        raise ValueError("voxel_size_mm must be positive")
    active = stiffness > 0
    if not bool(active.any().item()):
        raise ValueError("stiffness_gpa_xyz contains no active voxels")

    dims = tuple(int(v) for v in stiffness.shape)
    node_shape = (dims[0] + 1, dims[1] + 1, dims[2] + 1, 3)
    total_dofs = int(np.prod(node_shape))
    prescribed = torch.zeros(total_dofs, dtype=torch.bool, device=device)
    values = torch.zeros(total_dofs, dtype=dtype, device=device)
    _apply_dirichlet(
        prescribed,
        values,
        problem.fixed_displacement_coordinates,
        problem.fixed_displacement_values,
        node_shape=node_shape,
        torch=torch,
        device=device,
    )
    if problem.loaded_node_coordinates is not None:
        _apply_dirichlet(
            prescribed,
            values,
            problem.loaded_node_coordinates,
            problem.loaded_node_values,
            node_shape=node_shape,
            torch=torch,
            device=device,
        )

    free = ~prescribed
    if not bool(free.any().item()):
        raise ValueError("all displacement degrees of freedom are constrained")

    element_dofs = _element_dof_indices(dims, node_shape, torch=torch, device=device)
    ke = _hex8_stiffness(
        poisson_ratio=float(problem.poisson_ratio),
        spacing=float(problem.voxel_size_mm),
        dtype=dtype,
        torch=torch,
        device=device,
    )
    element_stiffness = stiffness.reshape(-1)
    active_elements = active.reshape(-1)
    element_dofs = element_dofs[active_elements]
    element_stiffness = element_stiffness[active_elements]

    fixed_u = values
    rhs_full = -_apply_operator(
        fixed_u,
        element_dofs=element_dofs,
        element_stiffness=element_stiffness,
        ke=ke,
        total_dofs=total_dofs,
        torch=torch,
    )
    rhs = rhs_full[free]

    def matvec_free(vector):
        full = torch.zeros(total_dofs, dtype=dtype, device=device)
        full[free] = vector
        return _apply_operator(
            full,
            element_dofs=element_dofs,
            element_stiffness=element_stiffness,
            ke=ke,
            total_dofs=total_dofs,
            torch=torch,
        )[free]

    max_iterations = (
        int(settings.max_iterations)
        if settings.max_iterations is not None
        else max(100, min(5000, int(free.sum().item()) * 2))
    )
    solution_free, diagnostics = _conjugate_gradient(
        matvec_free,
        rhs,
        tolerance=float(settings.tolerance),
        max_iterations=max_iterations,
        torch=torch,
    )
    displacement = fixed_u.clone()
    displacement[free] = solution_free
    residual = matvec_free(solution_free) - rhs
    forces = _apply_operator(
        displacement,
        element_dofs=element_dofs,
        element_stiffness=element_stiffness,
        ke=ke,
        total_dofs=total_dofs,
        torch=torch,
    )
    sed = _element_energy_density(
        displacement,
        element_dofs=element_dofs,
        element_stiffness=element_stiffness,
        ke=ke,
        dims=dims,
        active_elements=active_elements,
        voxel_volume=float(problem.voxel_size_mm) ** 3,
        torch=torch,
    )
    fields = {
        "displacements": displacement.reshape(node_shape).detach().cpu().numpy(),
        "forces": forces.reshape(node_shape).detach().cpu().numpy(),
        "sed": sed.detach().cpu().numpy(),
    }
    elapsed = perf_counter() - started
    residual_norm = float(torch.linalg.norm(residual).detach().cpu())
    diagnostics.update(
        {
            "backend": "parosol_torch",
            "device": str(device),
            "active_elements": int(active_elements.sum().item()),
            "free_dofs": int(free.sum().item()),
            "runtime_seconds": elapsed,
        }
    )
    return VoxelElasticityResult(
        fields={
            key: value
            for key, value in fields.items()
            if key in set(problem.requested_outputs) | {"displacements", "forces"}
            or key == "sed"
        },
        diagnostics=diagnostics,
        converged=bool(diagnostics["converged"]),
        iterations=int(diagnostics["iterations"]),
        residual_norm=residual_norm,
    )


def _select_device(torch, requested: str | None):
    if requested is None:
        return torch.device("cpu")
    normalized = requested.strip().lower()
    if normalized == "mps":
        return torch.device("mps")
    if normalized == "cuda":
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown torch device '{requested}'")


def _apply_dirichlet(
    prescribed,
    values,
    coordinates,
    coordinate_values,
    *,
    node_shape: tuple[int, int, int, int],
    torch,
    device,
) -> None:
    if coordinates is None:
        return
    coords_np = np.asarray(coordinates, dtype=np.int64)
    if coords_np.size == 0:
        return
    coords_np = coords_np.reshape(-1, 4)
    if coordinate_values is None:
        raise ValueError("displacement values are required when coordinates are set")
    vals_np = np.asarray(coordinate_values, dtype=np.float64).reshape(-1)
    if vals_np.size != coords_np.shape[0]:
        raise ValueError("displacement coordinate/value counts do not match")
    if np.any(coords_np[:, 0] < 0) or np.any(coords_np[:, 0] >= node_shape[0]):
        raise ValueError("x displacement coordinate outside node grid")
    if np.any(coords_np[:, 1] < 0) or np.any(coords_np[:, 1] >= node_shape[1]):
        raise ValueError("y displacement coordinate outside node grid")
    if np.any(coords_np[:, 2] < 0) or np.any(coords_np[:, 2] >= node_shape[2]):
        raise ValueError("z displacement coordinate outside node grid")
    if np.any(coords_np[:, 3] < 0) or np.any(coords_np[:, 3] >= 3):
        raise ValueError("component coordinate must be 0, 1, or 2")
    indices_np = np.ravel_multi_index(coords_np.T, node_shape)
    indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
    vals = torch.as_tensor(vals_np, dtype=values.dtype, device=device)
    prescribed[indices] = True
    values[indices] = vals


def _element_dof_indices(
    dims: tuple[int, int, int],
    node_shape: tuple[int, int, int, int],
    *,
    torch,
    device,
):
    node_offsets = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )
    elements = []
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                dofs = []
                for offset in node_offsets:
                    node = np.array([x, y, z], dtype=np.int64) + offset
                    for component in range(3):
                        dofs.append((*node.tolist(), component))
                elements.append(np.ravel_multi_index(np.asarray(dofs).T, node_shape))
    return torch.as_tensor(np.asarray(elements), dtype=torch.long, device=device)


def _hex8_stiffness(*, poisson_ratio: float, spacing: float, dtype, torch, device):
    if not (-1.0 < poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be between -1 and 0.5")
    nu = float(poisson_ratio)
    lam = nu / ((1 + nu) * (1 - 2 * nu))
    mu = 1.0 / (2 * (1 + nu))
    d = torch.tensor(
        [
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ],
        dtype=dtype,
        device=device,
    )
    points = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    ke = torch.zeros((24, 24), dtype=dtype, device=device)
    det_j = (spacing / 2.0) ** 3
    for xi in points:
        for eta in points:
            for zeta in points:
                gradients = _shape_gradients(float(xi), float(eta), float(zeta))
                b = torch.zeros((6, 24), dtype=dtype, device=device)
                for node, (dxi, deta, dzeta) in enumerate(gradients):
                    dx = dxi * 2.0 / spacing
                    dy = deta * 2.0 / spacing
                    dz = dzeta * 2.0 / spacing
                    col = 3 * node
                    b[0, col] = dx
                    b[1, col + 1] = dy
                    b[2, col + 2] = dz
                    b[3, col] = dy
                    b[3, col + 1] = dx
                    b[4, col + 1] = dz
                    b[4, col + 2] = dy
                    b[5, col] = dz
                    b[5, col + 2] = dx
                ke = ke + b.T @ d @ b * det_j
    return ke


def _shape_gradients(xi: float, eta: float, zeta: float):
    signs = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    gradients = []
    for sx, sy, sz in signs:
        gradients.append(
            (
                0.125 * sx * (1 + sy * eta) * (1 + sz * zeta),
                0.125 * sy * (1 + sx * xi) * (1 + sz * zeta),
                0.125 * sz * (1 + sx * xi) * (1 + sy * eta),
            )
        )
    return gradients


def _apply_operator(u, *, element_dofs, element_stiffness, ke, total_dofs: int, torch):
    ue = u[element_dofs]
    fe = torch.matmul(ue, ke.T) * element_stiffness[:, None]
    out = torch.zeros(total_dofs, dtype=u.dtype, device=u.device)
    out.index_add_(0, element_dofs.reshape(-1), fe.reshape(-1))
    return out


def _element_energy_density(
    u,
    *,
    element_dofs,
    element_stiffness,
    ke,
    dims: tuple[int, int, int],
    active_elements,
    voxel_volume: float,
    torch,
):
    ue = u[element_dofs]
    ku = torch.matmul(ue, ke.T) * element_stiffness[:, None]
    energy = 0.5 * torch.sum(ue * ku, dim=1) / voxel_volume
    sed = torch.zeros(int(np.prod(dims)), dtype=u.dtype, device=u.device)
    sed[active_elements] = energy
    return sed.reshape(dims)


def _conjugate_gradient(matvec, rhs, *, tolerance: float, max_iterations: int, torch):
    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    p = r.clone()
    rsold = torch.dot(r, r)
    rhs_norm = torch.linalg.norm(rhs)
    target = float(tolerance) * max(float(rhs_norm.detach().cpu()), 1.0)
    residual_norm = float(torch.sqrt(rsold).detach().cpu())
    if residual_norm <= target:
        return x, {
            "iterations": 0,
            "converged": True,
            "relative_residual": residual_norm
            / max(float(rhs_norm.detach().cpu()), 1.0),
        }
    iterations = 0
    converged = False
    for iterations in range(1, int(max_iterations) + 1):
        ap = matvec(p)
        denom = torch.dot(p, ap)
        if bool(denom == 0):
            break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * ap
        rsnew = torch.dot(r, r)
        residual_norm = float(torch.sqrt(rsnew).detach().cpu())
        if residual_norm <= target:
            converged = True
            rsold = rsnew
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    relative = residual_norm / max(float(rhs_norm.detach().cpu()), 1.0)
    return x, {
        "iterations": iterations,
        "converged": converged,
        "relative_residual": relative,
    }
