import math
import os
import tempfile
import unittest

from magcal.io import read_calibration_csv, read_raw_csv, write_corrected_csv
from magcal.linalg import RankDeficientError
from magcal.model import Sample, fit, leave_one_out, load_model, save_model, weighted_rmse

TRUE_MATRIX = [[1.10, 0.05, -0.02],
               [-0.03, 0.95, 0.04],
               [0.01, -0.06, 1.20]]
TRUE_OFFSET = [12.0, -7.5, 3.25]


def make_samples(noise=0.0):
    raws = [(10, 0, 0), (0, 10, 0), (0, 0, 10), (-10, 0, 0),
            (0, -10, 0), (0, 0, -10), (7, 7, 7), (-7, 7, -7),
            (5, -8, 3), (-4, 2, 9)]
    samples = []
    for i, raw in enumerate(raws):
        ref = tuple(
            TRUE_MATRIX[r][0] * raw[0] + TRUE_MATRIX[r][1] * raw[1]
            + TRUE_MATRIX[r][2] * raw[2] + TRUE_OFFSET[r]
            + noise * ((i + r) % 3 - 1)
            for r in range(3))
        samples.append(Sample(f"s{i}", tuple(float(v) for v in raw), ref,
                              1.0 + 0.1 * (i % 3)))
    return samples


class FitTests(unittest.TestCase):
    def test_recovers_cross_axis_coupling_and_offset(self):
        model = fit(make_samples())
        for r in range(3):
            for c in range(3):
                self.assertAlmostEqual(model.matrix[r][c], TRUE_MATRIX[r][c],
                                       places=9)
            self.assertAlmostEqual(model.offset[r], TRUE_OFFSET[r], places=9)

    def test_weighted_rmse_zero_on_exact_data(self):
        model = fit(make_samples())
        self.assertAlmostEqual(weighted_rmse(model, make_samples()), 0.0,
                               places=9)

    def test_rank_deficient_design_rejected(self):
        collinear = [Sample(f"p{i}", (float(i), 2.0 * i, -float(i)),
                            (1.0, 2.0, 3.0), 1.0) for i in range(6)]
        with self.assertRaises(RankDeficientError):
            fit(collinear)

    def test_too_few_samples_rejected(self):
        with self.assertRaises(ValueError):
            fit(make_samples()[:3])

    def test_loo_flags_degenerate_fold(self):
        samples = make_samples()
        samples.append(Sample("dup", samples[0].raw, samples[0].ref, 1.0))
        results = leave_one_out(samples)
        self.assertEqual(len(results), len(samples))
        self.assertTrue(all(note is None for _, _, note in results))
        degenerate = [Sample(f"c{i}", (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), 1.0)
                      for i in range(5)]
        results = leave_one_out(degenerate)
        self.assertTrue(all(note is not None for _, _, note in results))


class IoTests(unittest.TestCase):
    def test_model_roundtrip_and_apply_order(self):
        model = fit(make_samples())
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.json")
            save_model(model, model_path)
            loaded = load_model(model_path)
            self.assertEqual(loaded.matrix, model.matrix)
            self.assertEqual(loaded.offset, model.offset)
            self.assertEqual(loaded.output_unit, "uT")

            raw_csv = os.path.join(tmp, "raw.csv")
            with open(raw_csv, "w", encoding="utf-8") as fh:
                fh.write("id,raw_x,raw_y,raw_z\n"
                         "b,1,2,3\n"
                         "a,4,5,6\n")
            records = read_raw_csv(raw_csv)
            out_csv = os.path.join(tmp, "out.csv")
            write_corrected_csv(out_csv,
                                [(i, loaded.apply(r)) for i, r in records])
            with open(out_csv, encoding="utf-8") as fh:
                lines = fh.read().strip().splitlines()
            self.assertEqual(lines[0], "id,cal_x,cal_y,cal_z,magnitude")
            self.assertTrue(lines[1].startswith("b,"))
            self.assertTrue(lines[2].startswith("a,"))
            cal = loaded.apply((1.0, 2.0, 3.0))
            expected_mag = math.sqrt(sum(v * v for v in cal))
            self.assertAlmostEqual(float(lines[1].split(",")[4]),
                                   expected_mag, places=6)

    def test_calibration_csv_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("id,raw_x,raw_y,raw_z,ref_x,ref_y,ref_z,weight\n"
                         "x,1,2,3,4,5,6,nan\n")
            with self.assertRaises(ValueError):
                read_calibration_csv(path)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("id,raw_x,raw_y,raw_z,ref_x,ref_y,ref_z,weight\n"
                         "x,1,2,3,4,5,6,-1\n")
            with self.assertRaises(ValueError):
                read_calibration_csv(path)


if __name__ == "__main__":
    unittest.main()
