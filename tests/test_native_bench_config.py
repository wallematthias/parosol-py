from pathlib import Path

from parosol_py.native_bench import build_case_matrix


def test_build_case_matrix_contains_expected_benchmark_cases():
    cases = build_case_matrix(Path("/tmp/root"))
    names = [case["name"] for case in cases]
    assert names == [
        "mpi1_no_fields",
        "mpi1_sed",
        "mpi4_no_fields",
        "mpi4_sed",
    ]
