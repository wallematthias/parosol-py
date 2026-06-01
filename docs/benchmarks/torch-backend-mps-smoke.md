# Torch Backend MPS Smoke Benchmark

Date: 2026-06-01

Environment:

- macOS Apple Silicon
- Python: `/Users/matthias.walle/miniforge3/envs/parosol/bin/python`
- PyTorch: 2.12.0
- MPS: built and available

Command:

```bash
/Users/matthias.walle/miniforge3/envs/parosol/bin/python \
  scripts/benchmark_torch_backend.py
```

Results:

| Case | CPU time | MPS time | MPS speedup | Notes |
| --- | ---: | ---: | ---: | --- |
| `tiny_1x1x1` | 1.3019 s | 0.3964 s | 3.28x | One element; dominated by startup/setup |
| `trab_dense_3x3x3` | 0.0045 s | 0.0920 s | 0.05x | MPS launch overhead dominates |
| `trab_dense_5x5x5` | 0.0066 s | 0.1005 s | 0.07x | MPS launch overhead dominates |

Interpretation:

The MPS backend is functional on the tiny cube and a small TRAB_1240 dense crop,
but these small crops do not yet show useful acceleration. For tiny problems,
MPS overhead dominates. Meaningful GPU speedup will need larger problem sizes,
active-voxel batching, and preconditioning/chunking work before full HR-pQCT
volumes are practical.
