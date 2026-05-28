from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
import SimpleITK as sitk

from parosol_py import solve
from parosol_py.boundary_conditions import axial_compression
from parosol_py.runner import packaged_executable


FAIM_ROOT = Path("/Applications/Faim 10.0")


def test_axial_compression_constraints_match_faim_tiny_cube(tmp_path: Path):
    if not _has_faim():
        pytest.skip("FAIM 10.0 n88modelgenerator is not installed")

    labels_xyz = np.ones((3, 2, 2), dtype=np.int16)
    model_path = _run_faim_model(labels_xyz, tmp_path, solve_model=False)

    faim_constraints = _read_faim_axial_constraints(model_path)
    coords, values = axial_compression(
        np.ones(labels_xyz.shape, dtype=np.float32),
        axis="z",
        strain=-0.01,
    )
    parosol_constraints = {
        tuple(int(v) for v in coord): _as_zero_or_float(value)
        for coord, value in zip(coords, values)
    }

    assert set(parosol_constraints) == set(faim_constraints)
    assert parosol_constraints == pytest.approx(faim_constraints, abs=1e-7)


def test_parosol_sed_matches_faim_tiny_cube(tmp_path: Path):
    if not _has_faim():
        pytest.skip("FAIM 10.0 is not installed")
    if not packaged_executable().exists():
        pytest.skip(f"packaged ParOSol executable is not available: {packaged_executable()}")

    labels_xyz = np.ones((3, 3, 3), dtype=np.int16)
    model_path = _run_faim_model(labels_xyz, tmp_path / "faim", solve_model=True)
    faim_sed_xyz = _read_faim_sed_xyz(model_path)

    material_zyx = np.ones((3, 3, 3), dtype=np.float64) * 1000.0
    result = solve(
        material=material_zyx,
        spacing=(1.0, 1.0, 1.0),
        material_unit="MPa",
        test="axial",
        test_axis="z",
        strain=-0.01,
        outputs=("sed",),
        work_dir=tmp_path / "parosol",
        export_dir=tmp_path / "parosol" / "exports",
        tolerance=1e-8,
        level=2,
    )
    assert result.exported["sed"].exists()
    parosol_sed_zyx = sitk.GetArrayFromImage(sitk.ReadImage(str(result.exported["sed"])))
    parosol_sed_xyz = np.transpose(parosol_sed_zyx, (2, 1, 0))

    np.testing.assert_allclose(parosol_sed_xyz, faim_sed_xyz, rtol=1e-5, atol=5e-7)


def _has_faim() -> bool:
    return (
        (FAIM_ROOT / "setenv").exists()
        and (FAIM_ROOT / "bin" / "n88modelgenerator").exists()
        and (FAIM_ROOT / "bin" / "n88solver_slt").exists()
        and (FAIM_ROOT / "bin" / "n88derivedfields").exists()
    )


def _run_faim_model(labels_xyz: np.ndarray, work_dir: Path, *, solve_model: bool) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    labels_path = work_dir / "labels.mha"
    image = sitk.GetImageFromArray(np.transpose(labels_xyz, (2, 1, 0)))
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(image, str(labels_path))

    material_path = work_dir / "material.txt"
    material_path.write_text(
        "\n".join(
            [
                "MaterialDefinitions:",
                "    Material_001:",
                "        Type: LinearIsotropic",
                "        E: 1000",
                "        nu: 0.3",
                "MaterialTable:",
                "    1: Material_001",
                "",
            ]
        ),
        encoding="utf-8",
    )
    model_path = work_dir / "case.n88model"
    commands = [
        [
            str(FAIM_ROOT / "bin" / "n88modelgenerator"),
            str(labels_path),
            "--material_definitions",
            str(material_path),
            "--test",
            "axial",
            "--test_axis",
            "z",
            "--normal_strain",
            "-0.01",
            str(model_path),
        ]
    ]
    if solve_model:
        commands.extend(
            [
                [str(FAIM_ROOT / "bin" / "n88solver_slt"), str(model_path)],
                [str(FAIM_ROOT / "bin" / "n88derivedfields"), str(model_path)],
            ]
        )
    script = f"source {shlex.quote(str(FAIM_ROOT / 'setenv'))}; " + "; ".join(
        " ".join(shlex.quote(arg) for arg in command) for command in commands
    )
    completed = subprocess.run(
        ["bash", "-lc", script],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"FAIM command failed: {completed.stderr[-500:]}")
    return model_path


def _read_faim_axial_constraints(model_path: Path) -> dict[tuple[int, int, int, int], float]:
    with h5py.File(model_path, "r") as h5:
        assert h5["Parts/Part1/Elements/Hexahedrons/ElementNumber"].shape == (12,)
        node_coordinates = np.asarray(h5["Parts/Part1/NodeCoordinates"][:], dtype=np.float64)
        origin = node_coordinates.min(axis=0)
        constraints: dict[tuple[int, int, int, int], float] = {}
        for group_name in ("bottom_fixed", "top_fixed", "top_displacement"):
            group = h5[f"Constraints/{group_name}"]
            for node_number, sense, value in zip(
                group["NodeNumber"][:],
                group["Sense"][:],
                group["Value"][:],
            ):
                node_xyz = tuple(
                    int(v)
                    for v in np.rint(node_coordinates[int(node_number) - 1] - origin)
                )
                constraints[(*node_xyz, int(sense) - 1)] = _as_zero_or_float(value)
    return constraints


def _read_faim_sed_xyz(model_path: Path) -> np.ndarray:
    with h5py.File(model_path, "r") as h5:
        sed = np.asarray(
            h5["Solutions/Solution1/ElementValues/StrainEnergyDensity"][:],
            dtype=np.float64,
        )
        node_coordinates = np.asarray(h5["Parts/Part1/NodeCoordinates"][:], dtype=np.float64)
        node_numbers = (
            np.asarray(
                h5["Parts/Part1/Elements/Hexahedrons/NodeNumbers"][:],
                dtype=np.int64,
            )
            - 1
        )
    centers = node_coordinates[node_numbers].mean(axis=1)
    origin = centers.min(axis=0)
    spacing = _infer_spacing_from_centers(centers)
    indices = np.rint((centers - origin) / spacing).astype(np.int64)
    dimensions = tuple(int(v) for v in (indices.max(axis=0) + 1))
    dense = np.zeros(dimensions, dtype=np.float64)
    dense[indices[:, 0], indices[:, 1], indices[:, 2]] = sed
    return dense


def _infer_spacing_from_centers(centers: np.ndarray) -> np.ndarray:
    spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        unique = np.unique(np.round(centers[:, axis], decimals=8))
        diffs = np.diff(unique)
        spacing[axis] = diffs[diffs > 0].min(initial=1.0)
    return spacing


def _as_zero_or_float(value) -> float:
    value = float(value)
    if abs(value) < 1e-12:
        return 0.0
    return value
