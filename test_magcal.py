"""Tests for the magnetometer affine calibration tool (stdlib unittest)."""

import io
import json
import math
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

import magcal
from linalg import RankDeficientError, rank_of

TRUE_MATRIX = [
    [0.020, 0.0015, -0.0008],
    [-0.0012, 0.021, 0.0009],
    [0.0006, -0.0011, 0.019],
]
TRUE_OFFSET = [3.2, -1.7, 0.45]

RAW_SAMPLES = [
    [1200.0, -340.0, 800.0],
    [-950.0, 1500.0, 220.0],
    [300.0, 610.0, -1400.0],
    [-200.0, -1100.0, -650.0],
    [1750.0, 420.0, 90.0],
    [-1300.0, -720.0, 1150.0],
    [80.0, 55.0, -40.0],
    [640.0, -980.0, 1430.0],
]


def make_refs(raws):
    return [magcal.apply_model(TRUE_MATRIX, TRUE_OFFSET, r) for r in raws]


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(v) for v in row) + "\n")


class FitTests(unittest.TestCase):
    def test_recovers_cross_axis_matrix_and_offset(self):
        weights = [1.0 + 0.1 * i for i in range(len(RAW_SAMPLES))]
        matrix, offset = magcal.fit_affine(RAW_SAMPLES, make_refs(RAW_SAMPLES), weights)
        for a in range(3):
            for k in range(3):
                self.assertAlmostEqual(matrix[a][k], TRUE_MATRIX[a][k], places=9)
            self.assertAlmostEqual(offset[a], TRUE_OFFSET[a], places=6)

    def test_cross_terms_are_not_diagonal_only(self):
        matrix, _ = magcal.fit_affine(
            RAW_SAMPLES, make_refs(RAW_SAMPLES), [1.0] * len(RAW_SAMPLES)
        )
        off_diag = max(
            abs(matrix[a][k]) for a in range(3) for k in range(3) if a != k
        )
        self.assertGreater(off_diag, 1e-4)

    def test_direction_not_inverted(self):
        matrix, offset = magcal.fit_affine(
            RAW_SAMPLES, make_refs(RAW_SAMPLES), [1.0] * len(RAW_SAMPLES)
        )
        raw = [500.0, -250.0, 100.0]
        pred = magcal.apply_model(matrix, offset, raw)
        expect = magcal.apply_model(TRUE_MATRIX, TRUE_OFFSET, raw)
        for a in range(3):
            self.assertAlmostEqual(pred[a], expect[a], places=6)

    def test_rank_deficient_samples_rejected(self):
        # All raw vectors lie on a line -> design rank < 4.
        raws = [[float(i), 2.0 * i, -3.0 * i] for i in range(1, 7)]
        refs = make_refs(raws)
        with self.assertRaises(RankDeficientError):
            magcal.fit_affine(raws, refs, [1.0] * len(raws))

    def test_constant_column_rank_check(self):
        design = [[1.0, 2.0, 3.0, 1.0], [2.0, 4.0, 6.0, 2.0],
                  [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
        self.assertLess(rank_of(design), 4)

    def test_weighted_rms_definition(self):
        residuals = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
        weights = [2.0, 1.0]
        expect = math.sqrt((2.0 * 9.0 + 1.0 * 16.0) / 3.0)
        self.assertAlmostEqual(magcal.weighted_rms(residuals, weights), expect)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        refs = make_refs(RAW_SAMPLES)
        rows = [
            [f"s{i}", *RAW_SAMPLES[i], *refs[i], 1.0]
            for i in range(len(RAW_SAMPLES))
        ]
        self.cal_csv = os.path.join(self.dir, "cal.csv")
        write_csv(self.cal_csv, magcal.CALIBRATION_COLUMNS, rows)
        self.model = os.path.join(self.dir, "model.json")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = magcal.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_fit_apply_validate_roundtrip(self):
        code, out, _ = self.run_cli(["fit", self.cal_csv, self.model])
        self.assertEqual(code, 0)
        self.assertIn("weighted_rms_uT", out)
        self.assertIn("leave_one_out", out)
        with open(self.model, encoding="utf-8") as handle:
            model = json.load(handle)
        self.assertEqual(model["reference_unit"], "uT")
        self.assertEqual(len(model["matrix"]), 3)

        new_csv = os.path.join(self.dir, "new.csv")
        write_csv(new_csv, magcal.APPLY_INPUT_COLUMNS,
                  [["n1", 100.0, 200.0, 300.0], ["n2", -50.0, 60.0, 70.0]])
        code, out, _ = self.run_cli(["apply", self.model, new_csv])
        self.assertEqual(code, 0)
        lines = [l for l in out.strip().splitlines()]
        self.assertEqual(len(lines), 3)  # header + 2 rows, order preserved
        self.assertTrue(lines[1].startswith("n1,"))
        self.assertTrue(lines[2].startswith("n2,"))
        cal = magcal.apply_model(TRUE_MATRIX, TRUE_OFFSET, [100.0, 200.0, 300.0])
        got = [float(v) for v in lines[1].split(",")[1:4]]
        for a in range(3):
            self.assertAlmostEqual(got[a], cal[a], places=4)
        mag = float(lines[1].split(",")[4])
        self.assertAlmostEqual(mag, magcal.vector_norm(cal), places=4)

        code, out, _ = self.run_cli(["validate", self.model, self.cal_csv])
        self.assertEqual(code, 0)
        wrms_line = [l for l in out.splitlines() if l.startswith("weighted_rms_uT,")][0]
        self.assertLess(float(wrms_line.split(",")[1]), 1e-9)

    def test_fit_rejects_degenerate_csv_without_model(self):
        bad_csv = os.path.join(self.dir, "bad.csv")
        rows = [[f"d{i}", float(i), 2.0 * i, -i, 1.0, 2.0, 3.0, 1.0]
                for i in range(1, 6)]
        write_csv(bad_csv, magcal.CALIBRATION_COLUMNS, rows)
        code, _, err = self.run_cli(["fit", bad_csv, self.model])
        self.assertEqual(code, 2)
        self.assertIn("rank-deficient", err)
        self.assertFalse(os.path.exists(self.model))

    def test_fit_rejects_too_few_rows_and_bad_values(self):
        few_csv = os.path.join(self.dir, "few.csv")
        write_csv(few_csv, magcal.CALIBRATION_COLUMNS,
                  [["a", 1, 2, 3, 4, 5, 6, 1]] * 3)
        code, _, err = self.run_cli(["fit", few_csv, self.model])
        self.assertEqual(code, 1)
        self.assertIn("at least", err)

        nan_csv = os.path.join(self.dir, "nan.csv")
        rows = [[f"s{i}", *RAW_SAMPLES[i], *make_refs(RAW_SAMPLES)[i], 1.0]
                for i in range(5)]
        rows[0][1] = "nan"
        write_csv(nan_csv, magcal.CALIBRATION_COLUMNS, rows)
        code, _, err = self.run_cli(["fit", nan_csv, self.model])
        self.assertEqual(code, 1)
        self.assertIn("finite", err)

    def test_loo_marks_degenerate_fold(self):
        # 4 coplanar-ish points + 1 off-plane point: removing the off-plane
        # point leaves a rank-deficient fold.
        raws = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        refs = make_refs(raws)
        rows = [[f"p{i}", *raws[i], *refs[i], 1.0] for i in range(5)]
        csv_path = os.path.join(self.dir, "loo.csv")
        write_csv(csv_path, magcal.CALIBRATION_COLUMNS, rows)
        code, out, _ = self.run_cli(["fit", csv_path, self.model])
        self.assertEqual(code, 0)
        self.assertIn("degenerate_fold", out)


if __name__ == "__main__":
    unittest.main()
