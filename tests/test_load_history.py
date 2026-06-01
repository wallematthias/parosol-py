import json

import numpy as np

from parosol_py.cli import main
from parosol_py.load_history import estimate_load_history


def test_estimate_load_history_returns_non_negative_scaling():
    load_cases = [
        np.ones((2, 2, 2), dtype=np.float64) * 0.01,
        np.ones((2, 2, 2), dtype=np.float64) * 0.02,
    ]

    result = estimate_load_history(
        load_cases,
        np.ones((2, 2, 2), dtype=bool),
        target_average=0.02,
    )

    assert np.all(result.scaling_factors >= 0)
    assert result.mean > 0
    assert result.loading_history.shape == (2, 2, 2)


def test_cli_load_history_writes_summary_and_output(tmp_path):
    sed_a = np.ones((2, 2, 2), dtype=np.float64) * 0.01
    sed_b = np.ones((2, 2, 2), dtype=np.float64) * 0.02
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    np.save(tmp_path / "sed_a.npy", sed_a)
    np.save(tmp_path / "sed_b.npy", sed_b)
    np.save(tmp_path / "mask.npy", mask)

    assert (
        main(
            [
                "load-history",
                str(tmp_path / "sed_a.npy"),
                str(tmp_path / "sed_b.npy"),
                "--bone-mask",
                str(tmp_path / "mask.npy"),
                "--summary",
                str(tmp_path / "summary.json"),
                "-o",
                str(tmp_path / "history.npy"),
            ]
        )
        == 0
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "scaling_factors" in summary["load_history"]
    assert (tmp_path / "history.npy").exists()
