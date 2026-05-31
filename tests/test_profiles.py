from parosol_py import get_output_profile, get_solver_profile


def test_builtin_profiles_are_available():
    solver = get_solver_profile("legacy_axial")
    output = get_output_profile("quick_summary")

    assert solver.tolerance == 1e-6
    assert "sed" in solver.outputs
    assert output.export_fields is False
    assert output.image_fields == ()
