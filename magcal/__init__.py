"""Magnetometer affine calibration using only the Python standard library."""

from .model import (Model, Sample, fit, leave_one_out, load_model, residuals,
                    save_model, weighted_rmse)

__all__ = ["Model", "Sample", "fit", "leave_one_out", "load_model",
           "residuals", "save_model", "weighted_rmse"]
