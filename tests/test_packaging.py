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
        == "python scripts/stage_mpi_runtime.py"
    )
    assert "packaged_mpi_launcher" in pyproject["tool"]["cibuildwheel"]["test-command"]
    assert (
        pyproject["tool"]["cibuildwheel"]["macos"]["environment"][
            "MACOSX_DEPLOYMENT_TARGET"
        ]
        == "15.0"
    )
    assert pyproject["tool"]["scikit-build"]["wheel"]["packages"] == [
        "src/parosol_py",
        "src/parosol_torch",
    ]
    assert pyproject["tool"]["cibuildwheel"]["build"] == "cp311-* cp312-* cp313-*"
    assert pyproject["tool"]["scikit-build"]["cmake"]["version"] == ">=3.18"
    assert pyproject["project"]["optional-dependencies"]["torch"] == ["torch>=2.3"]
    assert "PAROSOL_MPI_RUNTIME openmpi msmpi" in cmake
    assert "DESTINATION parosol_py/bin" in cmake
    assert "install(PROGRAMS ${PAROSOL_MPI_RUNTIME_PROGRAMS}" in cmake


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
    assert "cibuildwheel" in str(wheels)
    assert "actions/upload-artifact" in str(wheels)
    assert "windows-latest" in str(wheels)
    assert "macos-15-intel" in str(wheels)
    assert "pypa/gh-action-pypi-publish" in str(wheels)
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
        assert "parosol_py/config_templates/profiles/xtremectii.yaml" in names
        assert "parosol_py/config_templates/profiles/vertebra.yaml" in names
