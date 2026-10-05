"""Small pure-stdlib linear algebra helpers (no third-party numeric libs)."""

from __future__ import annotations

PIVOT_RTOL = 1e-12


class RankDeficientError(Exception):
    """Raised when a matrix is not full rank within the pivot tolerance."""


def solve_square(matrix, rhs, rtol=PIVOT_RTOL):
    """Solve A x = b by Gaussian elimination with partial pivoting.

    Raises RankDeficientError if any pivot is smaller than
    ``rtol * max(abs(diagonal of the original matrix))``.  The tolerance is
    relative to the matrix scale so it is unit-agnostic; a pivot below it
    means the system has no numerically trustworthy unique solution.
    """
    n = len(matrix)
    a = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    scale = max((abs(a[i][i]) for i in range(n)), default=0.0)
    scale = max(scale, max(abs(v) for row in matrix for v in row))
    threshold = rtol * scale
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) <= threshold:
            raise RankDeficientError(
                f"matrix is rank deficient (pivot {abs(a[pivot][col]):.3e} "
                f"<= tolerance {threshold:.3e} at column {col})"
            )
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] * inv
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def weighted_normal_equations(rows, weights, targets):
    """Build A^T W A and A^T W y for design rows, weights and targets."""
    n = len(rows[0])
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, w, y in zip(rows, weights, targets):
        for i in range(n):
            wi = w * row[i]
            atb[i] += wi * y
            for j in range(i, n):
                ata[i][j] += wi * row[j]
    for i in range(n):
        for j in range(i):
            ata[i][j] = ata[j][i]
    return ata, atb
