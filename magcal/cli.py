"""Command line interface: fit, apply and validate subcommands."""
from __future__ import annotations

import argparse
import math
import sys

from .io import read_calibration_csv, read_raw_csv, write_corrected_csv
from .linalg import RankDeficientError
from .model import (fit, leave_one_out, load_model, residuals, save_model,
                    weighted_rmse)


def _fmt_vec(vec):
    return "(" + ", ".join(f"{v:+.6g}" for v in vec) + ")"


def _report(model, samples, out):
    print("per-row residuals (ref - prediction, uT):", file=out)
    for ident, resid in residuals(model, samples):
        norm = math.sqrt(sum(v * v for v in resid))
        print(f"  {ident}: residual {_fmt_vec(resid)} |resid|={norm:.6g}", file=out)
    print(f"weighted RMSE: {weighted_rmse(model, samples):.6g} uT", file=out)
    print("leave-one-out validation:", file=out)
    degenerate = 0
    for ident, resid, note in leave_one_out(samples):
        if resid is None:
            degenerate += 1
            print(f"  {ident}: DEGENERATE FOLD ({note})", file=out)
        else:
            norm = math.sqrt(sum(v * v for v in resid))
            print(f"  {ident}: residual {_fmt_vec(resid)} |resid|={norm:.6g}", file=out)
    if degenerate:
        print(f"  ({degenerate} degenerate fold(s) excluded from LOO summary)",
              file=out)


def cmd_fit(args):
    samples = read_calibration_csv(args.calibration_csv)
    try:
        model = fit(samples)
    except (RankDeficientError, ValueError) as exc:
        print(f"fit failed, no model written: {exc}", file=sys.stderr)
        return 2
    save_model(model, args.model_json)
    _report(model, samples, sys.stdout)
    print(f"model written to {args.model_json}")
    return 0


def cmd_apply(args):
    model = load_model(args.model_json)
    records = read_raw_csv(args.input_csv)
    corrected = [(ident, model.apply(raw)) for ident, raw in records]
    write_corrected_csv(args.output_csv, corrected)
    print(f"wrote {len(corrected)} corrected record(s) to {args.output_csv}")
    return 0


def cmd_validate(args):
    model = load_model(args.model_json)
    samples = read_calibration_csv(args.validation_csv)
    _report(model, samples, sys.stdout)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="magcal",
        description="Affine magnetometer calibration (pure standard library).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="fit calibration and save JSON model")
    p_fit.add_argument("calibration_csv")
    p_fit.add_argument("model_json")
    p_fit.set_defaults(func=cmd_fit)

    p_apply = sub.add_parser("apply", help="apply a saved model to new raw CSV")
    p_apply.add_argument("model_json")
    p_apply.add_argument("input_csv")
    p_apply.add_argument("output_csv")
    p_apply.set_defaults(func=cmd_apply)

    p_val = sub.add_parser("validate",
                           help="evaluate a saved model on an independent CSV")
    p_val.add_argument("model_json")
    p_val.add_argument("validation_csv")
    p_val.set_defaults(func=cmd_validate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
