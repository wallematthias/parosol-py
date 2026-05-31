from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_native_wheel_build_settings():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["build-system"]["build-backend"] == "scikit_build_core.build"
    assert "cibuildwheel" in pyproject["tool"]
    assert pyproject["tool"]["scikit-build"]["wheel"]["packages"] == ["src/parosol_py"]
    assert pyproject["tool"]["scikit-build"]["cmake"]["version"] == ">=3.18"


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
        "3.10",
        "3.11",
        "3.12",
        "3.13",
    ]
    assert "conda-incubator/setup-miniconda" in str(tests)
    assert "cibuildwheel" in str(wheels)
    assert "actions/upload-artifact" in str(wheels)
    assert "windows-latest" in str(wheels)
    assert "macos-15-intel" in str(wheels)
    assert "pypa/gh-action-pypi-publish" in str(wheels)
    assert "pypa/gh-action-pypi-publish" in str(publish)
