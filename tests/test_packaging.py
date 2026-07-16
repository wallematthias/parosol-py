import subprocess
from pathlib import Path
from zipfile import ZipFile

import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_native_wheel_build_settings():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert pyproject["build-system"]["build-backend"] == "scikit_build_core.build"
    assert pyproject["project"]["requires-python"] == ">=3.11,<3.14"
    assert "numpy>=1.24" in pyproject["project"]["dependencies"]
    assert not any(
        dependency.startswith("numpy") and "<2" in dependency
        for dependency in pyproject["project"]["dependencies"]
    )
    assert "cibuildwheel" in pyproject["tool"]
    assert (
        pyproject["tool"]["cibuildwheel"]["before-build"]
        == "python scripts/stage_mpi_runtime.py && python scripts/verify_packaged_mpi.py"
    )
    assert (
        pyproject["tool"]["cibuildwheel"]["test-command"]
        == "python -m parosol_py._wheel_smoke"
    )
    linux_cfg = pyproject["tool"]["cibuildwheel"]["linux"]
    assert linux_cfg["before-all"] == "bash scripts/install_linux_wheel_deps.sh"
    assert linux_cfg["environment"]["PAROSOL_OPENMPI_PREFIX"] == "/usr/lib64/openmpi"
    assert linux_cfg["environment"]["CMAKE_PREFIX_PATH"] == "/usr;/usr/lib64/openmpi"
    assert linux_cfg["environment"]["CMAKE_ARGS"] == (
        "-DMPI_CXX_COMPILER=/usr/lib64/openmpi/bin/mpicxx"
    )
    assert "PAROSOL_LINUX_CONDA_ROOT" not in linux_cfg["environment"]
    assert "CC" not in linux_cfg["environment"]
    assert "CXX" not in linux_cfg["environment"]
    assert "PATH" not in linux_cfg["environment"]
    assert "LD_LIBRARY_PATH" not in linux_cfg["environment"]
    assert "--plat manylinux_2_28_x86_64" in linux_cfg["repair-wheel-command"]
    assert "--exclude libmpi.so.40" in linux_cfg["repair-wheel-command"]
    linux_deps = (ROOT / "scripts" / "install_linux_wheel_deps.sh").read_text(
        encoding="utf-8"
    )
    assert "hdf5-devel" in linux_deps
    assert "openmpi-devel" in linux_deps
    assert "eigen3-devel" in linux_deps
    assert "conda" not in linux_deps
    assert "compilers" not in linux_deps
    assert (
        pyproject["tool"]["cibuildwheel"]["macos"]["environment"][
            "MACOSX_DEPLOYMENT_TARGET"
        ]
        == "15.0"
    )
    assert pyproject["tool"]["scikit-build"]["wheel"]["packages"] == [
        "src/parosol_py",
    ]
    assert pyproject["tool"]["cibuildwheel"]["build"] == "cp311-* cp312-* cp313-*"
    assert pyproject["tool"]["scikit-build"]["cmake"]["version"] == ">=3.18"
    assert "torch" not in pyproject["project"].get("optional-dependencies", {})
    assert "PAROSOL_MPI_RUNTIME openmpi msmpi" in cmake
    assert "DESTINATION parosol_py/bin" in cmake
    assert "install(PROGRAMS ${PAROSOL_MPI_RUNTIME_PROGRAMS}" in cmake
    assert "PAROSOL_WINDOWS_RUNTIME_DLLS" in cmake
    assert (
        pyproject["tool"]["cibuildwheel"]["windows"]["before-all"]
        == "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install_windows_wheel_deps.ps1"
    )
    windows_deps = (
        ROOT / "scripts" / "install_windows_wheel_deps.ps1"
    ).read_text(encoding="utf-8")
    assert "MaxAttempts" in windows_deps
    assert "Start-Sleep" in windows_deps
    assert "hdf5[cpp]:x64-windows" in windows_deps


def test_packaged_mpi_verifier_does_not_import_package_init():
    script = (ROOT / "scripts" / "verify_packaged_mpi.py").read_text(
        encoding="utf-8"
    )

    assert "importlib.util.spec_from_file_location" in script
    assert '_package_bin_dir = lambda: ROOT / "src" / "parosol_py" / "bin"' in script
    assert "from parosol_py.runner import" not in script


def test_github_workflows_build_test_and_wheels():
    tests = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
    wheels = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "build-wheels.yml").read_text()
    )
    publish = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "publish-from-run.yml").read_text()
    )

    assert "pull_request" in tests["on"]
    assert "push" in tests["on"]
    assert tests["jobs"]["test"]["strategy"]["matrix"]["python-version"] == [
        "3.11",
        "3.12",
        "3.13",
    ]
    coverage_setup_steps = [
        step
        for step in tests["jobs"]["coverage"]["steps"]
        if "conda-incubator/setup-miniconda" in step.get("uses", "")
    ]
    assert coverage_setup_steps[0]["with"]["python-version"] == "3.12"
    assert "conda-incubator/setup-miniconda" in str(tests)
    assert "pull_request" not in wheels["on"]
    assert wheels["on"]["push"] == {"tags": ["v*"]}
    assert "publish_existing" in wheels["on"]["workflow_dispatch"]["inputs"]["target"]["options"]
    assert "source_run_ids" in wheels["on"]["workflow_dispatch"]["inputs"]
    assert "cibuildwheel" in str(wheels)
    assert "actions/upload-artifact" in str(wheels)
    assert "windows-latest" in str(wheels)
    assert "macos-15-intel" in str(wheels)
    assert "Download artifacts from source runs" in str(wheels)
    assert "pypa/gh-action-pypi-publish" in str(wheels)
    assert "source_run_ids" in publish["on"]["workflow_dispatch"]["inputs"]
    assert "Download artifacts from each source run" in str(publish)
    assert "pypa/gh-action-pypi-publish" in str(publish)


def test_existing_wheel_artifacts_include_config_templates_when_present():
    assert (ROOT / "src" / "parosol_py" / "licenses" / "parosol_native_LICENSE.txt").is_file()

    wheels = sorted((ROOT / "dist").glob("*.whl")) + sorted(
        (ROOT / "dist-local-x86").glob("*.whl")
    )
    if not wheels:
        return

    for wheel in wheels:
        with ZipFile(wheel) as zf:
            names = set(zf.namelist())
        assert "parosol_py/config_templates/default.yaml" in names
        if "parosol_py/workflows/XtremeCTII.parosol-workflow" not in names:
            continue
        assert "parosol_py/workflows/XtremeCTII.parosol-workflow" in names
        assert "parosol_py/workflows/spine-compression.parosol-workflow" in names
        assert (
            "parosol_py/workflows/hip-sideways-fall-left.parosol-workflow"
            in names
        )
        assert (
            "parosol_py/workflows/hip-sideways-fall-right.parosol-workflow"
            in names
        )
        assert "parosol_py/workflows/vertebra.parosol-workflow" not in names
        assert "parosol_py/workflows/ct-hip-sideways-fall.parosol-workflow" not in names


def test_source_tree_does_not_include_generated_cache_files():
    forbidden_names = {"__pycache__", ".DS_Store"}
    tracked = subprocess.check_output(
        ["git", "ls-files", "src", "tests"], cwd=ROOT, text=True
    ).splitlines()
    offenders = [path for path in tracked if Path(path).name in forbidden_names]

    assert offenders == []


def test_readme_documents_workflow_case_model():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert ".parosol-workflow" in readme
    assert "parosol_case.yaml" in readme
    assert "--profile spine-compression" in readme
    assert "--profile hip-sideways-fall-left" in readme
    assert "--profile XtremeCTII" in readme
    assert "SlicerParOSol creates and edits" in readme
    assert "ct-spine-compression" not in readme
    assert "ct-hip-sideways-fall" not in readme
    assert "spine-batch" not in readme
    assert "proximal_femur" not in readme
