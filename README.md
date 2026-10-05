# Magnetometer Affine Calibration

Pure standard-library Python tool that fits an affine correction
`ref = M @ raw + offset` (full 3x3 cross-axis coupling matrix `M` plus a
3-vector zero offset) from bench reference measurements, saves the model as
JSON, and applies it to new recordings. No third-party numeric libraries.

## Commands

```
python -m magcal fit calibration.csv model.json      # fit + save model
python -m magcal apply model.json new.csv out.csv    # correct new samples
python -m magcal validate model.json holdout.csv     # evaluate only
```

`apply` only loads the saved model; it never refits. `validate` evaluates an
independent set without touching the coefficients.

## File formats

Calibration / validation CSV (header required, UTF-8, BOM tolerated):

```
id,raw_x,raw_y,raw_z,ref_x,ref_y,ref_z,weight
s1,10.2,0.1,-0.3,23.01,-7.80,3.35,1.0
```

- `id`: unique, non-empty record identifier.
- `raw_*`: raw magnetometer readings in instrument units.
- `ref_*`: reference field in microtesla (uT).
- `weight`: positive finite weight for that row.
- All numeric fields must be finite; at least 4 rows are required.

Apply-input CSV needs only `id,raw_x,raw_y,raw_z`. The output CSV is
`id,cal_x,cal_y,cal_z,magnitude` (uT), preserving input row order.

The JSON model stores `matrix` (3x3), `offset` (3-vector) and `units`
(`input`: instrument units, `output`: uT).

## Method and error definitions

- Objective: minimize `sum_i w_i * |ref_i - (M raw_i + b)|^2` over the 12
  coefficients, solved per output axis via the weighted normal equations
  with Gaussian elimination (partial pivoting).
- Rank check: the design matrix is `[raw_x raw_y raw_z 1]`; the fit is
  rejected when the normal matrix is not full rank. A pivot is treated as
  zero when `|pivot| <= 1e-12 * scale`, where `scale` is the largest
  absolute entry of the normal matrix (relative tolerance, unit-agnostic).
  Rank-deficient data is refused outright - no zero-padding, no
  regularization, and no model file is written on failure.
- Per-row residual: `ref_i - prediction_i` (vector, uT).
- Weighted RMSE: `sqrt(mean_i(w_i * |residual_i|^2))`, i.e. the square root
  of the mean weighted squared residual-vector norm.
- Leave-one-out: each row is predicted by a model refit on the remaining
  rows; folds whose reduced design is degenerate are marked
  `DEGENERATE FOLD` instead of producing a number.

## Tests

```
python -m unittest discover -s tests -v
```

Covers cross-axis coupling recovery, non-zero offset recovery, rank
deficiency rejection, degenerate LOO folds, CSV validation, and
order-preserving apply output.
