"""Affine magnetometer calibration model: ref ~= M @ raw + offset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .linalg import RankDeficientError, solve_square, weighted_normal_equations

MIN_SAMPLES = 4
MODEL_FORMAT = "magnetometer-affine-calibration/1"


@dataclass
class Sample:
    ident: str
    raw: tuple
    ref: tuple
    weight: float


@dataclass
class Model:
    matrix: list  # 3x3, maps raw instrument units to reference uT
    offset: list  # 3-vector in uT
    input_unit: str = "instrument"
    output_unit: str = "uT"

    def apply(self, raw):
        return tuple(
            self.matrix[r][0] * raw[0]
            + self.matrix[r][1] * raw[1]
            + self.matrix[r][2] * raw[2]
            + self.offset[r]
            for r in range(3)
        )

    def to_dict(self):
        return {
            "format": MODEL_FORMAT,
            "matrix": self.matrix,
            "offset": self.offset,
            "units": {"input": self.input_unit, "output": self.output_unit},
        }

    @classmethod
    def from_dict(cls, data):
        if data.get("format") != MODEL_FORMAT:
            raise ValueError(f"unsupported model format: {data.get('format')!r}")
        matrix = [[float(v) for v in row] for row in data["matrix"]]
        offset = [float(v) for v in data["offset"]]
        if len(matrix) != 3 or any(len(r) != 3 for r in matrix) or len(offset) != 3:
            raise ValueError("model must contain a 3x3 matrix and 3-vector offset")
        units = data.get("units", {})
        return cls(matrix, offset, units.get("input", "instrument"),
                   units.get("output", "uT"))


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def load_model(path):
    with open(path, "r", encoding="utf-8") as fh:
        return Model.from_dict(json.load(fh))


def _design_rows(samples):
    return [[s.raw[0], s.raw[1], s.raw[2], 1.0] for s in samples]


def fit(samples):
    """Weighted least-squares affine fit; refuses rank-deficient designs."""
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"need at least {MIN_SAMPLES} calibration records")
    rows = _design_rows(samples)
    weights = [s.weight for s in samples]
    matrix, offset = [], []
    for axis in range(3):
        ata, atb = weighted_normal_equations(
            rows, weights, [s.ref[axis] for s in samples])
        coef = solve_square(ata, atb)  # raises RankDeficientError
        matrix.append(coef[:3])
        offset.append(coef[3])
    return Model(matrix, offset)


def residuals(model, samples):
    """Per-row residual vectors (reference - prediction)."""
    out = []
    for s in samples:
        pred = model.apply(s.raw)
        out.append((s.ident, tuple(s.ref[i] - pred[i] for i in range(3))))
    return out


def weighted_rmse(model, samples):
    """sqrt(mean of weight * |residual vector|^2) over the samples."""
    total = 0.0
    for s in samples:
        pred = model.apply(s.raw)
        total += s.weight * sum((s.ref[i] - pred[i]) ** 2 for i in range(3))
    return math.sqrt(total / len(samples))


def leave_one_out(samples):
    """LOO cross-validation; degenerate folds are flagged, not hidden."""
    results = []
    for i, held in enumerate(samples):
        rest = samples[:i] + samples[i + 1:]
        try:
            fold_model = fit(rest)
        except (RankDeficientError, ValueError) as exc:
            results.append((held.ident, None, f"degenerate fold: {exc}"))
            continue
        pred = fold_model.apply(held.raw)
        resid = tuple(held.ref[a] - pred[a] for a in range(3))
        results.append((held.ident, resid, None))
    return results
