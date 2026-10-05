"""CSV reading and writing for calibration, application and validation."""
from __future__ import annotations

import csv
import math

from .model import Sample

CALIBRATION_COLUMNS = ("id", "raw_x", "raw_y", "raw_z",
                       "ref_x", "ref_y", "ref_z", "weight")
RAW_COLUMNS = ("id", "raw_x", "raw_y", "raw_z")
OUTPUT_COLUMNS = ("id", "cal_x", "cal_y", "cal_z", "magnitude")


def _finite(value, field, ident):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"record {ident!r}: {field} is not a number: {value!r}")
    if not math.isfinite(number):
        raise ValueError(f"record {ident!r}: {field} must be finite, got {value!r}")
    return number


def _require_columns(path, columns, required):
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s): {missing}")


def read_calibration_csv(path):
    """Read id, raw xyz, reference xyz and positive weight per row."""
    samples = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        _require_columns(path, reader.fieldnames or [], CALIBRATION_COLUMNS)
        seen = set()
        for row in reader:
            ident = (row["id"] or "").strip()
            if not ident:
                raise ValueError(f"{path}: record with empty id")
            if ident in seen:
                raise ValueError(f"{path}: duplicate id {ident!r}")
            seen.add(ident)
            raw = tuple(_finite(row[c], c, ident) for c in ("raw_x", "raw_y", "raw_z"))
            ref = tuple(_finite(row[c], c, ident) for c in ("ref_x", "ref_y", "ref_z"))
            weight = _finite(row["weight"], "weight", ident)
            if weight <= 0.0:
                raise ValueError(f"record {ident!r}: weight must be positive")
            samples.append(Sample(ident, raw, ref, weight))
    return samples


def read_raw_csv(path):
    """Read id and raw xyz rows for the apply command."""
    records = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        _require_columns(path, reader.fieldnames or [], RAW_COLUMNS)
        for row in reader:
            ident = (row["id"] or "").strip()
            raw = tuple(_finite(row[c], c, ident) for c in ("raw_x", "raw_y", "raw_z"))
            records.append((ident, raw))
    return records


def write_corrected_csv(path, records):
    """Write corrected triples and vector magnitude, preserving input order."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(OUTPUT_COLUMNS)
        for ident, vec in records:
            magnitude = math.sqrt(sum(v * v for v in vec))
            writer.writerow([ident] + [f"{v:.9g}" for v in vec]
                            + [f"{magnitude:.9g}"])
