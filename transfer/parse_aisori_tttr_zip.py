#!/usr/bin/env python3
"""Parse AISORI TTTR ZIP export into a typed CSV.

Expected input ZIP contains files like:
- wr*.txt       (data rows)
- fld*.txt      (field descriptors)
- statlist*.txt (station list)

For TTTR daily exports with "Все" request fields enabled, data rows are expected
to have 9 tokens:
1) WMO station index
2) year
3) month
4) day
5) quality flag
6) Tmin (C)
7) Tmean (C)
8) Tmax (C)
9) precipitation (mm)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zip", required=True, help="Path to AISORI wr*.zip archive.")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output parsed CSV path. Default: <zip_stem>_parsed.csv in the same folder.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Default: <output_csv>.summary.json",
    )
    return parser.parse_args()


def infer_data_entry(names: list[str]) -> str:
    candidates = [name for name in names if re.match(r"^wr\d+a\d+\.txt$", Path(name).name)]
    if not candidates:
        candidates = [name for name in names if Path(name).name.startswith("wr") and Path(name).suffix.lower() == ".txt"]
    if not candidates:
        raise RuntimeError("No wr*.txt data entry found inside ZIP archive.")
    return candidates[0]


def safe_int(value: str) -> int:
    return int(value.strip())


def safe_float(value: str) -> float:
    return float(value.strip().replace(",", "."))


def main() -> int:
    args = parse_args()
    in_zip = Path(args.input_zip).expanduser().resolve()
    if not in_zip.exists():
        raise SystemExit(f"Input ZIP not found: {in_zip}")

    if args.output_csv:
        out_csv = Path(args.output_csv).expanduser().resolve()
    else:
        out_csv = in_zip.with_name(f"{in_zip.stem}_parsed.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.summary_json:
        summary_json = Path(args.summary_json).expanduser().resolve()
    else:
        summary_json = out_csv.with_suffix(out_csv.suffix + ".summary.json")

    rows_total = 0
    rows_written = 0
    rows_bad = 0
    station_set: set[str] = set()
    date_min: str | None = None
    date_max: str | None = None

    with zipfile.ZipFile(in_zip, "r") as zf:
        entry = infer_data_entry(zf.namelist())
        with zf.open(entry, "r") as fin, out_csv.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(
                fout,
                fieldnames=[
                    "Date",
                    "station",
                    "quality_flag",
                    "T_min",
                    "T_hydromet",
                    "T_max",
                    "P_mm",
                ],
            )
            writer.writeheader()

            for raw in fin:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                rows_total += 1
                parts = re.split(r"\s+", line)
                if len(parts) != 9:
                    rows_bad += 1
                    continue
                try:
                    station = parts[0]
                    y = safe_int(parts[1])
                    m = safe_int(parts[2])
                    d = safe_int(parts[3])
                    qf = safe_int(parts[4])
                    t_min = safe_float(parts[5])
                    t_mean = safe_float(parts[6])
                    t_max = safe_float(parts[7])
                    p_mm = safe_float(parts[8])
                    dt = date(y, m, d).isoformat()
                except Exception:
                    rows_bad += 1
                    continue

                writer.writerow(
                    {
                        "Date": dt,
                        "station": station,
                        "quality_flag": qf,
                        "T_min": t_min,
                        "T_hydromet": t_mean,
                        "T_max": t_max,
                        "P_mm": p_mm,
                    }
                )
                rows_written += 1
                station_set.add(station)
                if date_min is None or dt < date_min:
                    date_min = dt
                if date_max is None or dt > date_max:
                    date_max = dt

    summary = {
        "input_zip": str(in_zip),
        "output_csv": str(out_csv),
        "summary_json": str(summary_json),
        "rows_total": rows_total,
        "rows_written": rows_written,
        "rows_bad": rows_bad,
        "stations": len(station_set),
        "date_min": date_min,
        "date_max": date_max,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

