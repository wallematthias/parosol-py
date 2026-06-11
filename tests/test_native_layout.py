from parosol_py.native_layout import choose_cpu_grid


def test_choose_cpu_grid_prefers_long_axis_for_rectangular_volume():
    dims = choose_cpu_grid(global_shape_xyz=(331, 502, 168), mpi_size=4)
    assert sorted(dims) == [1, 2, 2]
    assert dims[1] >= dims[2]


def test_choose_cpu_grid_uses_single_rank_for_serial_case():
    assert choose_cpu_grid(global_shape_xyz=(100, 120, 80), mpi_size=1) == (1, 1, 1)
