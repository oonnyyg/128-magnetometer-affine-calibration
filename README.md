# Magnetometer Affine Calibration

Pure Python standard library tool (no third-party numeric packages) that
fits an affine magnetometer calibration

    reference_uT ~= matrix(3x3) @ raw_counts + offset(3)

including inter-axis (cross-coupling) terms and zero offsets, then applies
the saved model to new recordings.

## Commands

    python magcal.py fit CALIBRATION.csv model.json
    python magcal.py apply model.json NEW_SAMPLES.csv > corrected.csv
    python magcal.py validate model.json VALIDATION.csv

- `fit` solves the weighted least squares problem
  `min sum_i w_i * |matrix @ raw_i + offset - ref_i|^2`, writes the JSON
  model, prints per-row residuals, the weighted RMS error, and
  leave-one-out (LOO) validation residuals. Each LOO fold refits on the
  remaining rows only; folds whose reduced design matrix is rank deficient
  are marked `degenerate_fold`.
- `apply` only loads the JSON model (never refits) and outputs
  `id,cal_x_uT,cal_y_uT,cal_z_uT,magnitude_uT` in input record order.
- `validate` evaluates a saved model on an independent CSV (same format as
  the calibration file). It is evaluation only and never adjusts
  coefficients.

## CSV formats

Calibration / validation CSV header:

    id,raw_x,raw_y,raw_z,ref_x,ref_y,ref_z,weight

Apply-input CSV header:

    id,raw_x,raw_y,raw_z

- `id` must be unique and non-empty (calibration file).
- Raw values are in instrument reading units (counts); reference values are
  in microtesla (uT).
- All numeric fields must be finite; `weight` must be positive.
- At least 4 calibration records are required.

## Model file

JSON with `matrix` (3x3, uT per count), `offset` (3, uT), and unit metadata
(`raw_unit`, `reference_unit`, `output_unit`). The correction direction is
`corrected_uT = matrix @ raw_counts + offset`.

## Rank check and numerical tolerance

Before solving, the design matrix `[raw_x raw_y raw_z 1]` must have full
column rank 4. Rank is determined by Gaussian elimination with partial
pivoting; a pivot is treated as zero when
`|pivot| <= 1e-12 * max|column entry|` (relative tolerance `RANK_TOL` in
`linalg.py`). Rank-deficient data is rejected outright — no zero-padding or
regularization is used to fake a unique calibration — and no model file is
written on failure (exit code 2).

## Error metrics

- Per-row residual: `ref_i - prediction_i`, per axis and its vector norm.
- Weighted RMS: `sqrt( sum_i w_i * |residual_i|^2 / sum_i w_i )`, i.e. the
  weighted mean of squared residual vector norms, then square root.
- LOO residual: norm of the residual of the held-out row under a model
  refit without that row.

## Tests

    python -m unittest test_magcal

Covers cross-axis coupling recovery, non-zero offset recovery, correction
direction, rank-deficient sample rejection, LOO degenerate folds, and the
weighted RMS definition.
