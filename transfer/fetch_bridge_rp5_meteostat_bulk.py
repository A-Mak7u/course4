from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from meteostat import config as meteostat_config
from meteostat.api import daily


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Массовая выгрузка RP5-like (Meteostat) для станций Росгидромета с resume и отбором station-set для bridge"
    )
    parser.add_argument(
        "--station-csv",
        default="data/rosgidromet/aisori/aisori_station_catalog.csv",
        help="CSV-каталог станций (по умолчанию AISORI station catalog).",
    )
    parser.add_argument("--station-col", default="station", help="Колонка station id в station-csv.")
    parser.add_argument(
        "--station-ids-file",
        default=None,
        help="Опциональный TXT со station id (по одному в строке), чтобы ограничить выборку.",
    )
    parser.add_argument("--start-year", type=int, default=2013)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument(
        "--hydromet-csv",
        default="data/rosgidromet/aisori/aisori_tttr_daily_2010_2025_merged.csv",
        help="Merged daily CSV Росгидромета (Date/station/T_hydromet).",
    )
    parser.add_argument(
        "--work-dir",
        default="data/rosgidromet/bridge_inputs/meteostat_bulk_2013_2023",
        help="Папка для per-station CSV и служебных файлов.",
    )
    parser.add_argument(
        "--output-rp5-csv",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_daily_2013_2023_allstations.csv",
        help="Итоговый объединённый RP5-like CSV.",
    )
    parser.add_argument(
        "--output-overlap-csv",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_vs_hydromet_overlap_2013_2023_allstations.csv",
        help="Итоговый overlap CSV.",
    )
    parser.add_argument(
        "--output-stats-csv",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_station_overlap_stats_2013_2023.csv",
        help="CSV со station-wise статистикой overlap.",
    )
    parser.add_argument(
        "--output-selected-csv",
        default="transfer/hydromet_bridge_station_ids_selected.csv",
        help="CSV со станциями, прошедшими фильтр отбора.",
    )
    parser.add_argument(
        "--output-selected-txt",
        default="transfer/hydromet_bridge_station_ids_selected.txt",
        help="TXT со станциями, прошедшими фильтр отбора.",
    )
    parser.add_argument("--summary-json", default=None, help="JSON-сводка (по умолчанию <output-stats-csv>.summary.json)")
    parser.add_argument("--min-hydromet-rows", type=int, default=1500, help="Минимум строк T_hydromet в 2013-2023 для участия.")
    parser.add_argument("--min-overlap-rows", type=int, default=1200, help="Минимум строк overlap для попадания в selected station-set.")
    parser.add_argument("--min-overlap-years", type=int, default=6, help="Минимум уникальных лет overlap для selected station-set.")
    parser.add_argument(
        "--exact-equal-ratio-max",
        type=float,
        default=0.98,
        help="Максимально допустимая доля точных совпадений T_rp5 == T_hydromet в selected station-set.",
    )
    parser.add_argument("--limit-stations", type=int, default=None, help="Ограничить число станций для debug/smoke.")
    parser.add_argument("--force-refetch", action="store_true", help="Игнорировать per-station cache и качать заново.")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_station_id(value: object) -> str:
    sid = "".join(ch for ch in str(value).strip() if ch.isdigit())
    return sid


def read_station_ids(station_csv: str, station_col: str, station_ids_file: str | None) -> list[str]:
    df = pd.read_csv(station_csv)
    if station_col not in df.columns:
        raise RuntimeError(f"Колонка {station_col!r} не найдена в {station_csv}")
    ids = [normalize_station_id(v) for v in df[station_col].tolist()]
    ids = [v for v in ids if v]

    if station_ids_file:
        allowed: list[str] = []
        text = Path(station_ids_file).read_text(encoding="utf-8")
        for line in text.splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            sid = normalize_station_id(token.split()[0])
            if sid:
                allowed.append(sid)
        allowed_set = set(allowed)
        ids = [sid for sid in ids if sid in allowed_set]

    unique_ids: list[str] = []
    seen: set[str] = set()
    for sid in ids:
        if sid in seen:
            continue
        seen.add(sid)
        unique_ids.append(sid)
    return unique_ids


def fetch_one_station(station_id: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str]:
    try:
        df = daily.daily(station_id, start=date(start_year, 1, 1), end=date(end_year, 12, 31)).fetch().reset_index()
    except Exception as exc:
        return pd.DataFrame(columns=["Date", "station", "T_rp5"]), f"fetch_error: {exc!r}"

    if df.empty:
        return pd.DataFrame(columns=["Date", "station", "T_rp5"]), "empty"
    if "temp" not in df.columns:
        return pd.DataFrame(columns=["Date", "station", "T_rp5"]), "missing_temp_col"

    out = df.rename(columns={"time": "Date", "temp": "T_rp5"}).copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["station"] = station_id
    out["T_rp5"] = pd.to_numeric(out["T_rp5"], errors="coerce")
    out = out.dropna(subset=["Date", "T_rp5"]).copy()
    out = out[["Date", "station", "T_rp5"]].sort_values("Date").drop_duplicates(["Date"], keep="last").reset_index(drop=True)
    if out.empty:
        return out, "no_valid_rows"
    return out, "ok"


def build_hydromet_subset(path: str, station_ids: list[str], start_year: int, end_year: int) -> pd.DataFrame:
    hyd = pd.read_csv(path, usecols=["Date", "station", "T_hydromet"]).copy()
    hyd["Date"] = pd.to_datetime(hyd["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    hyd["station"] = hyd["station"].astype(str).map(normalize_station_id)
    hyd["T_hydromet"] = pd.to_numeric(hyd["T_hydromet"], errors="coerce")
    hyd = hyd.dropna(subset=["Date", "station", "T_hydromet"])
    hyd = hyd[
        (hyd["Date"] >= f"{start_year}-01-01")
        & (hyd["Date"] <= f"{end_year}-12-31")
        & hyd["station"].isin(station_ids)
    ].copy()
    return hyd


def main() -> None:
    args = parse_args()
    meteostat_config.block_large_requests = False

    station_ids = read_station_ids(args.station_csv, args.station_col, args.station_ids_file)
    if args.limit_stations:
        station_ids = station_ids[: args.limit_stations]
    if not station_ids:
        raise RuntimeError("Пустой station-set после фильтрации.")

    hyd = build_hydromet_subset(args.hydromet_csv, station_ids=station_ids, start_year=args.start_year, end_year=args.end_year)
    hyd_counts = hyd.groupby("station").size().rename("hydromet_rows").reset_index()
    if args.min_hydromet_rows > 0:
        keep = set(hyd_counts[hyd_counts["hydromet_rows"] >= args.min_hydromet_rows]["station"].astype(str))
        station_ids = [sid for sid in station_ids if sid in keep]
    if not station_ids:
        raise RuntimeError("После фильтра min_hydromet_rows station-set пуст.")

    work_dir = Path(args.work_dir)
    per_station_dir = work_dir / "per_station"
    per_station_dir.mkdir(parents=True, exist_ok=True)

    fetch_rows: list[dict[str, object]] = []
    for i, sid in enumerate(station_ids, start=1):
        out_csv = per_station_dir / f"{sid}.csv"
        if out_csv.exists() and not args.force_refetch:
            cached = pd.read_csv(out_csv)
            fetch_rows.append(
                {
                    "station": sid,
                    "status": "cached",
                    "rows": int(len(cached)),
                    "date_min": str(cached["Date"].min()) if len(cached) else None,
                    "date_max": str(cached["Date"].max()) if len(cached) else None,
                }
            )
            print(f"[{i}/{len(station_ids)}] {sid}: cached ({len(cached)} rows)")
            continue

        station_df, status = fetch_one_station(sid, start_year=args.start_year, end_year=args.end_year)
        if status == "ok":
            station_df.to_csv(out_csv, index=False)
        fetch_rows.append(
            {
                "station": sid,
                "status": status,
                "rows": int(len(station_df)),
                "date_min": str(station_df["Date"].min()) if len(station_df) else None,
                "date_max": str(station_df["Date"].max()) if len(station_df) else None,
            }
        )
        print(f"[{i}/{len(station_ids)}] {sid}: {status} ({len(station_df)} rows)")

    fetch_df = pd.DataFrame(fetch_rows).sort_values(["status", "station"]).reset_index(drop=True)
    fetch_df.to_csv(work_dir / "fetch_manifest.csv", index=False)

    ready_ids = fetch_df[fetch_df["rows"] > 0]["station"].astype(str).tolist()
    rp5_frames: list[pd.DataFrame] = []
    for sid in ready_ids:
        p = per_station_dir / f"{sid}.csv"
        if p.exists():
            rp5_frames.append(pd.read_csv(p))

    if not rp5_frames:
        raise RuntimeError("После выгрузки нет ни одной станции с валидными RP5-like рядами.")

    rp5 = pd.concat(rp5_frames, ignore_index=True)
    rp5["Date"] = pd.to_datetime(rp5["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    rp5["station"] = rp5["station"].astype(str).map(normalize_station_id)
    rp5["T_rp5"] = pd.to_numeric(rp5["T_rp5"], errors="coerce")
    rp5 = rp5.dropna(subset=["Date", "station", "T_rp5"]).copy()
    rp5 = rp5.sort_values(["station", "Date"]).drop_duplicates(["station", "Date"], keep="last").reset_index(drop=True)

    out_rp5 = Path(args.output_rp5_csv)
    out_rp5.parent.mkdir(parents=True, exist_ok=True)
    rp5.to_csv(out_rp5, index=False)

    hyd = hyd[hyd["station"].isin(set(rp5["station"].astype(str)))].copy()
    overlap = rp5.merge(hyd, on=["Date", "station"], how="inner").dropna(subset=["T_hydromet"])
    overlap["delta"] = overlap["T_rp5"] - overlap["T_hydromet"]
    overlap["abs_delta"] = overlap["delta"].abs()
    overlap["year"] = pd.to_datetime(overlap["Date"], errors="coerce").dt.year
    overlap = overlap.dropna(subset=["year"]).copy()
    overlap["year"] = overlap["year"].astype(int)

    out_overlap = Path(args.output_overlap_csv)
    out_overlap.parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(out_overlap, index=False)

    rows: list[dict[str, object]] = []
    for sid, g in overlap.groupby("station"):
        rmse = float(np.sqrt(np.mean(np.square(g["delta"]))))
        rows.append(
            {
                "station": sid,
                "overlap_rows": int(len(g)),
                "overlap_years": int(g["year"].nunique()),
                "date_min": str(g["Date"].min()),
                "date_max": str(g["Date"].max()),
                "baseline_mae": float(g["abs_delta"].mean()),
                "baseline_rmse": rmse,
                "baseline_bias": float(g["delta"].mean()),
                "exact_equal_ratio": float((g["abs_delta"] == 0).mean()),
                "abs_delta_median": float(g["abs_delta"].median()),
                "abs_delta_max": float(g["abs_delta"].max()),
            }
        )
    stats_df = pd.DataFrame(rows).sort_values(["overlap_rows", "station"], ascending=[False, True]).reset_index(drop=True)
    out_stats = Path(args.output_stats_csv)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_stats, index=False)

    selected = stats_df[
        (stats_df["overlap_rows"] >= args.min_overlap_rows)
        & (stats_df["overlap_years"] >= args.min_overlap_years)
        & (stats_df["exact_equal_ratio"] <= args.exact_equal_ratio_max)
    ].copy()
    selected_ids = selected["station"].astype(str).tolist()

    out_sel_csv = Path(args.output_selected_csv)
    out_sel_txt = Path(args.output_selected_txt)
    out_sel_csv.parent.mkdir(parents=True, exist_ok=True)
    out_sel_txt.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"station_id": selected_ids}).to_csv(out_sel_csv, index=False)
    out_sel_txt.write_text("\n".join(selected_ids) + ("\n" if selected_ids else ""), encoding="utf-8")

    summary_path = Path(args.summary_json) if args.summary_json else out_stats.with_suffix(out_stats.suffix + ".summary.json")
    summary = {
        "station_csv": str(Path(args.station_csv).resolve()),
        "station_col": args.station_col,
        "requested_station_count": int(len(station_ids)),
        "fetch_ok_or_cached_count": int((fetch_df["rows"] > 0).sum()),
        "fetch_failed_or_empty_count": int((fetch_df["rows"] == 0).sum()),
        "rp5_rows_total": int(len(rp5)),
        "rp5_station_count": int(rp5["station"].nunique()),
        "overlap_rows_total": int(len(overlap)),
        "overlap_station_count": int(overlap["station"].nunique()),
        "overlap_date_min": str(overlap["Date"].min()) if len(overlap) else None,
        "overlap_date_max": str(overlap["Date"].max()) if len(overlap) else None,
        "selected_station_count": int(len(selected_ids)),
        "selected_station_ids": selected_ids,
        "thresholds": {
            "min_hydromet_rows": int(args.min_hydromet_rows),
            "min_overlap_rows": int(args.min_overlap_rows),
            "min_overlap_years": int(args.min_overlap_years),
            "exact_equal_ratio_max": float(args.exact_equal_ratio_max),
        },
        "paths": {
            "work_dir": str(work_dir.resolve()),
            "fetch_manifest_csv": str((work_dir / "fetch_manifest.csv").resolve()),
            "output_rp5_csv": str(out_rp5.resolve()),
            "output_overlap_csv": str(out_overlap.resolve()),
            "output_stats_csv": str(out_stats.resolve()),
            "output_selected_csv": str(out_sel_csv.resolve()),
            "output_selected_txt": str(out_sel_txt.resolve()),
        },
    }
    save_json(summary_path, summary)

    print(f"Saved manifest: {work_dir / 'fetch_manifest.csv'}")
    print(f"Saved RP5-like csv: {out_rp5}")
    print(f"Saved overlap csv: {out_overlap}")
    print(f"Saved station stats: {out_stats}")
    print(f"Saved selected station set: {out_sel_txt} ({len(selected_ids)} stations)")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
