from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from meteostat import config as meteostat_config
from meteostat.api import daily

from pipeline_common import save_json


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Сбор источника T_rp5 из Meteostat для RP5->Росгидромет bridge")
    parser.add_argument(
        "--station-ids",
        default=None,
        help="Список station id через запятую (например: 27947,28900,34163,34391).",
    )
    parser.add_argument(
        "--station-ids-file",
        default="transfer/hydromet_available_station_ids.txt",
        help="TXT-файл с station id (по одному в строке).",
    )
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument(
        "--output-csv",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_daily_2013_2023.csv",
        help="Итоговый CSV с колонками Date,station,T_rp5.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="JSON со сводкой. По умолчанию: <output-csv>.summary.json",
    )
    parser.add_argument(
        "--hydromet-csv",
        default="data/rosgidromet/aisori/aisori_tttr_daily_2010_2025_merged.csv",
        help="Опциональный CSV Росгидромета для overlap-сводки.",
    )
    return parser


def parse_station_ids(raw: str | None, ids_file: str | None) -> list[str]:
    ids: list[str] = []
    if raw:
        ids.extend(part.strip() for part in raw.split(","))
    if ids_file:
        path = Path(ids_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                ids.append(token.split()[0])

    out: list[str] = []
    seen: set[str] = set()
    for token in ids:
        sid = "".join(ch for ch in token if ch.isdigit())
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def fetch_station_daily(station_id: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    df = daily.daily(station_id, start=start_dt, end=end_dt).fetch().reset_index()
    if df.empty:
        return pd.DataFrame(columns=["Date", "station", "T_rp5"])
    if "temp" not in df.columns:
        raise RuntimeError(f"Meteostat daily for station {station_id} has no 'temp' column")
    out = df.rename(columns={"time": "Date", "temp": "T_rp5"}).copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out["station"] = station_id
    out["T_rp5"] = pd.to_numeric(out["T_rp5"], errors="coerce")
    return out[["Date", "station", "T_rp5"]]


def build_overlap_probe(
    rp5_df: pd.DataFrame,
    hydromet_csv: str,
    station_ids: list[str],
    start_year: int,
    end_year: int,
) -> dict[str, object] | None:
    path = Path(hydromet_csv)
    if not path.exists():
        return None

    hyd = pd.read_csv(path, usecols=["Date", "station", "T_hydromet"]).copy()
    hyd["Date"] = pd.to_datetime(hyd["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    hyd["station"] = hyd["station"].astype(str).str.strip()
    hyd["T_hydromet"] = pd.to_numeric(hyd["T_hydromet"], errors="coerce")
    hyd = hyd.dropna(subset=["Date", "station", "T_hydromet"])
    hyd = hyd[
        (hyd["Date"] >= f"{start_year}-01-01")
        & (hyd["Date"] <= f"{end_year}-12-31")
        & hyd["station"].isin(station_ids)
    ].copy()

    overlap = rp5_df.merge(hyd, on=["Date", "station"], how="inner")
    if overlap.empty:
        return {
            "hydromet_csv": str(path.resolve()),
            "overlap_rows": 0,
            "overlap_stations": 0,
        }

    diff = overlap["T_rp5"] - overlap["T_hydromet"]
    abs_diff = diff.abs()
    return {
        "hydromet_csv": str(path.resolve()),
        "overlap_rows": int(len(overlap)),
        "overlap_stations": int(overlap["station"].nunique()),
        "overlap_date_min": str(overlap["Date"].min()),
        "overlap_date_max": str(overlap["Date"].max()),
        "abs_delta_mean": float(abs_diff.mean()),
        "abs_delta_median": float(abs_diff.median()),
        "abs_delta_max": float(abs_diff.max()),
        "exact_equal_ratio": float((abs_diff == 0).mean()),
        "overlap_rows_by_station": {
            str(k): int(v) for k, v in overlap.groupby("station").size().sort_values(ascending=False).items()
        },
    }


def main() -> None:
    args = make_parser().parse_args()
    station_ids = parse_station_ids(args.station_ids, args.station_ids_file)
    if not station_ids:
        raise RuntimeError("Пустой список station id. Укажите --station-ids или --station-ids-file")

    meteostat_config.block_large_requests = False
    start_dt = date(args.start_year, 1, 1)
    end_dt = date(args.end_year, 12, 31)

    frames: list[pd.DataFrame] = []
    rows_per_station: dict[str, int] = {}
    for sid in station_ids:
        station_df = fetch_station_daily(sid, start_dt=start_dt, end_dt=end_dt)
        rows_per_station[sid] = int(len(station_df))
        if not station_df.empty:
            frames.append(station_df)

    if not frames:
        raise RuntimeError("Meteostat вернул пустые ряды по всем station id")

    out = pd.concat(frames, ignore_index=True)
    out["T_rp5"] = pd.to_numeric(out["T_rp5"], errors="coerce")
    out = out.dropna(subset=["Date", "station", "T_rp5"]).copy()
    out = out.sort_values(["station", "Date"]).drop_duplicates(["station", "Date"], keep="last").reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary_path = Path(args.summary_json) if args.summary_json else output_csv.with_suffix(output_csv.suffix + ".summary.json")
    summary = {
        "output_csv": str(output_csv.resolve()),
        "requested_station_ids": station_ids,
        "requested_station_count": int(len(station_ids)),
        "rows_total": int(len(out)),
        "stations_with_data": int(out["station"].nunique()),
        "date_min": str(out["Date"].min()) if not out.empty else None,
        "date_max": str(out["Date"].max()) if not out.empty else None,
        "rows_by_station_raw_fetch": rows_per_station,
        "rows_by_station_final": {
            str(k): int(v) for k, v in out.groupby("station").size().sort_values(ascending=False).items()
        },
        "missing_station_ids": [sid for sid in station_ids if sid not in set(out["station"].astype(str))],
    }
    overlap_probe = build_overlap_probe(
        rp5_df=out,
        hydromet_csv=args.hydromet_csv,
        station_ids=station_ids,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    if overlap_probe is not None:
        summary["hydromet_overlap_probe"] = overlap_probe
    save_json(summary_path, summary)

    print(f"Saved rp5-like Meteostat CSV: {output_csv}")
    print(f"Saved summary: {summary_path}")
    print(f"Rows={len(out)}, stations={out['station'].nunique()}, date={out['Date'].min()}..{out['Date'].max()}")


if __name__ == "__main__":
    main()
