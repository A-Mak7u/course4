from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from pipeline_common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сбор overlap-таблицы RP5 vs Росгидромет")
    parser.add_argument(
        "--rp5-csv",
        default="final_2013_2023_T_ERA5_LST_daynight.csv",
        help="CSV с RP5-рядом (должны быть station/date/T).",
    )
    parser.add_argument("--rp5-date-col", default="Date")
    parser.add_argument("--rp5-station-col", default="Cod")
    parser.add_argument("--rp5-temp-col", default="T")
    parser.add_argument(
        "--hydromet-glob",
        default="data/rosgidromet/aisori/wr*_parsed.csv",
        help="Глоб-паттерн parsed CSV Росгидромета.",
    )
    parser.add_argument(
        "--output-merged-hydromet",
        default="data/rosgidromet/aisori/aisori_tttr_daily_2010_2025_merged.csv",
        help="Итоговый объединённый CSV Росгидромета.",
    )
    parser.add_argument(
        "--output-overlap",
        default="data/rosgidromet/bridge_inputs/rp5_vs_hydromet_overlap_2013_2023.csv",
        help="Итоговый overlap CSV для bridge.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="JSON со сводкой. По умолчанию: <output-overlap>.summary.json",
    )
    parser.add_argument(
        "--reference-stations-csv",
        default=None,
        help="Опциональный CSV для проверки пересечения station-кодов (например, целевой набор региона).",
    )
    parser.add_argument(
        "--reference-station-col",
        default="Cod",
        help="Колонка station id в reference-stations-csv.",
    )
    parser.add_argument(
        "--dedupe-keep",
        choices=("first", "last"),
        default="last",
        help="Какой источник оставлять при дублях Date+station после merge.",
    )
    return parser.parse_args()


def _normalize_date_col(series: pd.Series, utc: bool) -> pd.Series:
    if utc:
        dt = pd.to_datetime(series, utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.date.astype("string")


def _load_hydromet(paths: list[str], keep: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        if "Date" not in df.columns or "station" not in df.columns or "T_hydromet" not in df.columns:
            raise RuntimeError(f"Файл {path} не содержит обязательные колонки Date/station/T_hydromet")
        df = df.copy()
        df["source_file"] = Path(path).name
        df["Date"] = _normalize_date_col(df["Date"], utc=False)
        df["station"] = df["station"].astype(str).str.strip()
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Date", "station"]).copy()
    out = out.sort_values(["Date", "station", "source_file"])
    out = out.drop_duplicates(subset=["Date", "station"], keep=keep)
    out = out.reset_index(drop=True)
    return out


def _load_rp5(path: str, date_col: str, station_col: str, temp_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[date_col, station_col, temp_col]).copy()
    df = df.rename(columns={date_col: "Date", station_col: "station", temp_col: "T_rp5"})
    df["Date"] = _normalize_date_col(df["Date"], utc=True)
    df["station"] = df["station"].astype(str).str.strip()
    df["T_rp5"] = pd.to_numeric(df["T_rp5"], errors="coerce")
    return df.dropna(subset=["Date", "station", "T_rp5"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    hydromet_paths = sorted(glob.glob(args.hydromet_glob))
    if not hydromet_paths:
        raise RuntimeError(f"Не найдены hydromet CSV по паттерну: {args.hydromet_glob}")

    hydromet = _load_hydromet(hydromet_paths, keep=args.dedupe_keep)
    rp5 = _load_rp5(
        path=args.rp5_csv,
        date_col=args.rp5_date_col,
        station_col=args.rp5_station_col,
        temp_col=args.rp5_temp_col,
    )

    overlap = rp5.merge(hydromet[["Date", "station", "T_hydromet"]], on=["Date", "station"], how="inner")
    overlap["T_hydromet"] = pd.to_numeric(overlap["T_hydromet"], errors="coerce")
    overlap = overlap.dropna(subset=["T_hydromet"]).reset_index(drop=True)
    overlap["delta_rp5_minus_hydromet"] = overlap["T_rp5"] - overlap["T_hydromet"]

    out_hyd = Path(args.output_merged_hydromet)
    out_hyd.parent.mkdir(parents=True, exist_ok=True)
    hydromet.to_csv(out_hyd, index=False)

    out_overlap = Path(args.output_overlap)
    out_overlap.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(out_overlap, index=False)

    summary_path = Path(args.summary_json) if args.summary_json else out_overlap.with_suffix(out_overlap.suffix + ".summary.json")
    abs_delta = overlap["delta_rp5_minus_hydromet"].abs()
    per_station_counts = overlap.groupby("station").size().sort_values(ascending=False)
    reference_report: dict[str, object] | None = None
    ref_path = Path(args.reference_stations_csv) if args.reference_stations_csv else None
    if ref_path is not None and ref_path.exists():
        ref = pd.read_csv(ref_path)
        if args.reference_station_col not in ref.columns:
            raise RuntimeError(
                f"Колонка {args.reference_station_col!r} не найдена в reference CSV: {args.reference_stations_csv}"
            )
        ref_ids = set(ref[args.reference_station_col].astype(str).str.strip())
        hyd_ids = set(hydromet["station"].astype(str))
        inter_ids = sorted(ref_ids & hyd_ids, key=lambda x: int(x))
        missing_ids = sorted(ref_ids - hyd_ids, key=lambda x: int(x))
        reference_report = {
            "reference_csv": str(ref_path.resolve()),
            "reference_station_col": args.reference_station_col,
            "reference_station_count": int(len(ref_ids)),
            "intersection_count": int(len(inter_ids)),
            "intersection_stations": inter_ids,
            "missing_in_hydromet_count": int(len(missing_ids)),
            "missing_in_hydromet_stations": missing_ids,
        }

    summary = {
        "rp5_csv": str(Path(args.rp5_csv).resolve()),
        "hydromet_glob": args.hydromet_glob,
        "hydromet_files_used": [str(Path(p).resolve()) for p in hydromet_paths],
        "output_merged_hydromet": str(out_hyd.resolve()),
        "output_overlap": str(out_overlap.resolve()),
        "dedupe_keep": args.dedupe_keep,
        "hydromet_rows": int(len(hydromet)),
        "hydromet_stations": int(hydromet["station"].nunique()),
        "hydromet_date_min": str(hydromet["Date"].min()) if not hydromet.empty else None,
        "hydromet_date_max": str(hydromet["Date"].max()) if not hydromet.empty else None,
        "rp5_rows": int(len(rp5)),
        "rp5_stations": int(rp5["station"].nunique()),
        "rp5_date_min": str(rp5["Date"].min()) if not rp5.empty else None,
        "rp5_date_max": str(rp5["Date"].max()) if not rp5.empty else None,
        "overlap_rows": int(len(overlap)),
        "overlap_stations": int(overlap["station"].nunique()),
        "overlap_date_min": str(overlap["Date"].min()) if not overlap.empty else None,
        "overlap_date_max": str(overlap["Date"].max()) if not overlap.empty else None,
        "abs_delta_mean": float(abs_delta.mean()) if len(abs_delta) else None,
        "abs_delta_median": float(abs_delta.median()) if len(abs_delta) else None,
        "abs_delta_max": float(abs_delta.max()) if len(abs_delta) else None,
        "exact_equal_ratio": float((abs_delta == 0).mean()) if len(abs_delta) else None,
        "overlap_rows_by_station": {str(k): int(v) for k, v in per_station_counts.items()},
        "reference_station_intersection": reference_report,
    }
    save_json(summary_path, summary)

    print(f"Saved merged hydromet: {out_hyd}")
    print(f"Saved overlap: {out_overlap}")
    print(f"Saved summary: {summary_path}")
    print(
        "Overlap stats:",
        f"rows={summary['overlap_rows']},",
        f"stations={summary['overlap_stations']},",
        f"abs_delta_mean={summary['abs_delta_mean']}",
        f"exact_equal_ratio={summary['exact_equal_ratio']}",
    )


if __name__ == "__main__":
    main()
