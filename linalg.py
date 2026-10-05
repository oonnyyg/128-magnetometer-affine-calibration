"""Pure-stdlib linear algebra helpers for weighted least squares.

Matrices are plain lists of lists of floats.
"""

# Relative pivot tolerance for rank determination and singularity checks.
# A pivot is treated as zero when |pivot| <= RANK_TOL * (largest |entry|
# in its column during elimination). Being relative, the criterion is
# insensitive to the absolute scale of the measurements.
RANK_TOL = 1e-12


class RankDeficientError(Exception):
    """Raised when the design matrix does not have full column rank."""


def rank_of(matrix, tol=RANK_TOL):
    """Return the numerical rank of a matrix via Gaussian elimination."""
    rows = [list(r) for r in matrix]
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    rank = 0
    col = 0
    while rank < n_rows and col < n_cols:
        pivot = max(range(rank, n_rows), key=lambda r: abs(rows[r][col]))
        col_max = max(abs(rows[r][col]) for r in range(n_rows))
        if col_max == 0.0 or abs(rows[pivot][col]) <= tol * col_max:
            col += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_val = rows[rank][col]
        for r in range(rank + 1, n_rows):
            factor = rows[r][col] / pivot_val
            if factor:
                for c in range(col, n_cols):
                    rows[r][c] -= factor * rows[rank][c]
        rank += 1
        col += 1
    return rank


def solve_square(matrix, rhs, tol=RANK_TOL):
    """Solve matrix @ x = rhs for a square matrix.

    Raises RankDeficientError when the matrix is numerically singular.
    """
    n = len(matrix)
    aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        col_max = max(abs(aug[r][col]) for r in range(n))
        if col_max == 0.0 or abs(aug[pivot][col]) <= tol * col_max:
            raise RankDeficientError("normal equations are singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for r in range(col + 1, n):
            factor = aug[r][col] / pivot_val
            if factor:
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = aug[row][n]
        for c in range(row + 1, n):
            acc -= aug[row][c] * x[c]
        x[row] = acc / aug[row][row]
    return x


def weighted_least_squares(design, targets, weights):
    """Solve min sum_i w_i * (design[i] @ beta - targets[i])^2.

    Raises RankDeficientError when the design matrix lacks full column
    rank; no zero-padding or regularization fakes a unique solution.
    """
    n_cols = len(design[0])
    r = rank_of(design)
    if r < n_cols:
        raise RankDeficientError(
            f"design matrix rank {r} < {n_cols} columns"
        )
    normal = [[0.0] * n_cols for _ in range(n_cols)]
    rhs = [0.0] * n_cols
    for row, target, w in zip(design, targets, weights):
        for a in range(n_cols):
            rhs[a] += w * row[a] * target
            for b in range(a, n_cols):
                normal[a][b] += w * row[a] * row[b]
    for a in range(n_cols):
        for b in range(a):
            normal[a][b] = normal[b][a]
    return solve_square(normal, rhs)
