"""Magnetometer affine calibration tool (pure Python standard library).

Model: reference_uT ~= matrix @ raw_counts + offset
Commands: fit, apply, validate. See README.md for file formats.
"""

import argparse
import csv
import json
import math
import sys

from linalg import RankDeficientError, weighted_least_squares

MODEL_VERSION = 1
MIN_CALIBRATION_ROWS = 4

CALIBRATION_COLUMNS = ["id", "raw_x", "raw_y", "raw_z", "ref_x", "ref_y", "ref_z", "weight"]
APPLY_INPUT_COLUMNS = ["id", "raw_x", "raw_y", "raw_z"]


class DataError(Exception):
    """Raised for malformed or insufficient CSV input."""


def _parse_float(text, field, row_no):
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise DataError(f"row {row_no}: field '{field}' is not a number: {text!r}")
    if not math.isfinite(value):
        raise DataError(f"row {row_no}: field '{field}' must be finite, got {text!r}")
    return value


def _read_rows(path, required_columns):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in required_columns if c not in (reader.fieldnames or [])]
        if missing:
            raise DataError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise DataError(f"{path}: no data rows")
    return rows


def load_calibration_csv(path):
    """Return (ids, raw_vectors, ref_vectors, weights) from a calibration CSV."""
    rows = _read_rows(path, CALIBRATION_COLUMNS)
    if len(rows) < MIN_CALIBRATION_ROWS:
        raise DataError(
            f"{path}: need at least {MIN_CALIBRATION_ROWS} records, got {len(rows)}"
        )
    ids, raws, refs, weights = [], [], [], []
    seen = set()
    for i, row in enumerate(rows, start=2):
        rid = (row["id"] or "").strip()
        if not rid:
            raise DataError(f"row {i}: empty id")
        if rid in seen:
            raise DataError(f"row {i}: duplicate id {rid!r}")
        seen.add(rid)
        ids.append(rid)
        raws.append([_parse_float(row[c], c, i) for c in ("raw_x", "raw_y", "raw_z")])
        refs.append([_parse_float(row[c], c, i) for c in ("ref_x", "ref_y", "ref_z")])
        weight = _parse_float(row["weight"], "weight", i)
        if weight <= 0.0:
            raise DataError(f"row {i}: weight must be positive, got {weight}")
        weights.append(weight)
    return ids, raws, refs, weights


def load_raw_csv(path):
    """Return (ids, raw_vectors) from an apply-input CSV."""
    rows = _read_rows(path, APPLY_INPUT_COLUMNS)
    ids, raws = [], []
    for i, row in enumerate(rows, start=2):
        rid = (row["id"] or "").strip()
        if not rid:
            raise DataError(f"row {i}: empty id")
        ids.append(rid)
        raws.append([_parse_float(row[c], c, i) for c in ("raw_x", "raw_y", "raw_z")])
    return ids, raws


def fit_affine(raws, refs, weights):
    """Fit matrix (3x3) and offset (3,) so matrix @ raw + offset ~= ref.

    Raises RankDeficientError when the augmented design matrix
    [raw_x raw_y raw_z 1] is rank deficient.
    """
    design = [[r[0], r[1], r[2], 1.0] for r in raws]
    matrix, offset = [], []
    for axis in range(3):
        targets = [ref[axis] for ref in refs]
        beta = weighted_least_squares(design, targets, weights)
        matrix.append(beta[:3])
        offset.append(beta[3])
    return matrix, offset


def apply_model(matrix, offset, raw):
    return [
        sum(matrix[a][k] * raw[k] for k in range(3)) + offset[a]
        for a in range(3)
    ]


def vector_norm(vec):
    return math.sqrt(sum(v * v for v in vec))


def weighted_rms(residuals, weights):
    """sqrt( sum_i w_i * |res_i|^2 / sum_i w_i )."""
    total_w = sum(weights)
    acc = sum(w * sum(c * c for c in res) for res, w in zip(residuals, weights))
    return math.sqrt(acc / total_w)


def save_model(path, matrix, offset):
    model = {
        "version": MODEL_VERSION,
        "type": "affine",
        "raw_unit": "counts",
        "reference_unit": "uT",
        "output_unit": "uT",
        "matrix": matrix,
        "offset": offset,
        "expression": "corrected_uT = matrix @ raw_counts + offset",
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2)
        handle.write("\n")


def load_model(path):
    with open(path, encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("version") != MODEL_VERSION or model.get("type") != "affine":
        raise DataError(f"{path}: unsupported model file")
    matrix = model["matrix"]
    offset = model["offset"]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix) or len(offset) != 3:
        raise DataError(f"{path}: malformed matrix/offset")
    for value in [v for row in matrix for v in row] + list(offset):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise DataError(f"{path}: non-finite model coefficient")
    return matrix, offset


def cmd_fit(args):
    ids, raws, refs, weights = load_calibration_csv(args.calibration_csv)
    try:
        matrix, offset = fit_affine(raws, refs, weights)
    except RankDeficientError as exc:
        print(f"fit failed: rank-deficient design matrix ({exc}); "
              "no model written", file=sys.stderr)
        return 2
    save_model(args.model, matrix, offset)

    residuals = []
    for raw, ref in zip(raws, refs):
        pred = apply_model(matrix, offset, raw)
        residuals.append([ref[a] - pred[a] for a in range(3)])
    wrms = weighted_rms(residuals, weights)

    out = csv.writer(sys.stdout)
    out.writerow(["id", "residual_x_uT", "residual_y_uT", "residual_z_uT",
                  "residual_norm_uT"])
    for rid, res in zip(ids, residuals):
        out.writerow([rid] + [f"{v:.6g}" for v in res] + [f"{vector_norm(res):.6g}"])
    print(f"weighted_rms_uT,{wrms:.6g}")

    print("leave_one_out:")
    out.writerow(["id", "status", "loo_residual_norm_uT"])
    for i, rid in enumerate(ids):
        sub_raws = raws[:i] + raws[i + 1:]
        sub_refs = refs[:i] + refs[i + 1:]
        sub_weights = weights[:i] + weights[i + 1:]
        try:
            m2, o2 = fit_affine(sub_raws, sub_refs, sub_weights)
        except RankDeficientError:
            out.writerow([rid, "degenerate_fold", ""])
            continue
        pred = apply_model(m2, o2, raws[i])
        res = [refs[i][a] - pred[a] for a in range(3)]
        out.writerow([rid, "ok", f"{vector_norm(res):.6g}"])
    print(f"model written to {args.model}")
    return 0


def cmd_apply(args):
    matrix, offset = load_model(args.model)
    ids, raws = load_raw_csv(args.input_csv)
    out = csv.writer(sys.stdout)
    out.writerow(["id", "cal_x_uT", "cal_y_uT", "cal_z_uT", "magnitude_uT"])
    for rid, raw in zip(ids, raws):
        cal = apply_model(matrix, offset, raw)
        out.writerow([rid] + [f"{v:.6g}" for v in cal] + [f"{vector_norm(cal):.6g}"])
    return 0


def cmd_validate(args):
    matrix, offset = load_model(args.model)
    ids, raws, refs, weights = load_calibration_csv(args.validation_csv)
    residuals = []
    for raw, ref in zip(raws, refs):
        pred = apply_model(matrix, offset, raw)
        residuals.append([ref[a] - pred[a] for a in range(3)])
    wrms = weighted_rms(residuals, weights)
    out = csv.writer(sys.stdout)
    out.writerow(["id", "residual_x_uT", "residual_y_uT", "residual_z_uT",
                  "residual_norm_uT"])
    for rid, res in zip(ids, residuals):
        out.writerow([rid] + [f"{v:.6g}" for v in res] + [f"{vector_norm(res):.6g}"])
    print(f"weighted_rms_uT,{wrms:.6g}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit affine model and report residuals/LOO")
    p_fit.add_argument("calibration_csv")
    p_fit.add_argument("model", help="output JSON model path")
    p_fit.set_defaults(func=cmd_fit)

    p_apply = sub.add_parser("apply", help="apply a saved model to new raw CSV")
    p_apply.add_argument("model")
    p_apply.add_argument("input_csv")
    p_apply.set_defaults(func=cmd_apply)

    p_val = sub.add_parser("validate", help="evaluate a model on an independent CSV")
    p_val.add_argument("model")
    p_val.add_argument("validation_csv")
    p_val.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
