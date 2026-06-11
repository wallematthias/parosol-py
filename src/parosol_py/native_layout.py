from __future__ import annotations

from itertools import permutations, product


def _factorizations(n: int) -> list[tuple[int, int, int]]:
    triples: set[tuple[int, int, int]] = set()
    for a in range(1, n + 1):
        if n % a != 0:
            continue
        rem = n // a
        for b in range(1, rem + 1):
            if rem % b != 0:
                continue
            c = rem // b
            triples.add((a, b, c))
    return sorted(triples)


def _surface_cost(shape_xyz: tuple[int, int, int], grid_xyz: tuple[int, int, int]) -> float:
    sx, sy, sz = (float(v) for v in shape_xyz)
    px, py, pz = (float(v) for v in grid_xyz)
    bx, by, bz = sx / px, sy / py, sz / pz
    # Cost proxy: total internal cut area plus a mild imbalance penalty.
    cut_area = (
        max(px - 1.0, 0.0) * by * bz
        + max(py - 1.0, 0.0) * bx * bz
        + max(pz - 1.0, 0.0) * bx * by
    )
    block_max = max(bx, by, bz)
    block_min = min(bx, by, bz)
    imbalance = block_max / block_min if block_min > 0 else float("inf")
    return cut_area * imbalance


def choose_cpu_grid(
    global_shape_xyz: tuple[int, int, int], mpi_size: int
) -> tuple[int, int, int]:
    if mpi_size <= 1:
        return (1, 1, 1)
    candidates = _factorizations(int(mpi_size))
    best: tuple[int, int, int] | None = None
    best_cost = float("inf")
    best_order_key: tuple[float, float, float] | None = None
    for candidate in candidates:
        for perm in set(permutations(candidate)):
            cost = _surface_cost(global_shape_xyz, perm)
            order_key = tuple(-float(s) / float(p) for s, p in zip(global_shape_xyz, perm))
            if cost < best_cost or (cost == best_cost and order_key < best_order_key):
                best = perm
                best_cost = cost
                best_order_key = order_key
    assert best is not None
    return best
