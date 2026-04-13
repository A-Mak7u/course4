from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подготовка policy-наборов overlap: expanded(min_n) и control(selected stations)"
    )
    parser.add_argument(
        "--input-csv",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_vs_hydromet_overlap_2013_2023_allstations.csv",
        help="Источник overlap CSV.",
    )
    parser.add_argument(
        "--selected-stations-file",
        default="transfer/hydromet_bridge_station_ids_selected.txt",
        help="TXT с selected station id (по одному на строку).",
    )
    parser.add_argument("--year-start", type=int, default=2013)
    parser.add_argument("--year-end", type=int, default=2023)
    parser.add_argument("--min-station-rows", type=int, default=10)
    parser.add_argument(
        "--output-expanded-csv",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_vs_hydromet_overlap_expanded_min10_2013_2023.csv",
    )
    parser.add_argument(
        "--output-control-csv",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_vs_hydromet_overlap_control_selected125_2013_2023.csv",
    )
    parser.add_argument(
        "--output-expanded-stations-txt",
        default="transfer/hydromet_bridge_station_ids_expanded_min10_2013_2023.txt",
    )
    parser.add_argument(
        "--output-expanded-stations-csv",
        default="transfer/hydromet_bridge_station_ids_expanded_min10_2013_2023.csv",
    )
    parser.add_argument(
        "--summary-json",
        default="data/rosgidromet/bridge_inputs/rp5_meteostat_overlap_policy_sets_2013_2023_min10.summary.json",
    )
    return parser.parse_args()


def read_selected_station_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        ids.append(token)
    return ids


def main() -> None:
    args = parse_args()
    src = Path(args.input_csv)
    if not src.exists():
        raise FileNotFoundError(f"Input overlap CSV not found: {src}")

    df = pd.read_csv(src).copy()
    required = {"Date", "station", "T_rp5", "T_hydromet"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Input CSV missing required columns: {sorted(missing)}")

    df["station"] = df["station"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["T_rp5"] = pd.to_numeric(df["T_rp5"], errors="coerce")
    df["T_hydromet"] = pd.to_numeric(df["T_hydromet"], errors="coerce")
    df = df.dropna(subset=["Date", "station", "T_rp5", "T_hydromet"]).copy()
    df = df[(df["Date"].dt.year >= args.year_start) & (df["Date"].dt.year <= args.year_end)].copy()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df = df.sort_values(["Date", "station"]).reset_index(drop=True)

    counts = df.groupby("station").size().rename("rows").reset_index()
    expanded_stations = (
        counts[counts["rows"] >= int(args.min_station_rows)]["station"].astype(str).sort_values().tolist()
    )
    expanded = df[df["station"].isin(set(expanded_stations))].copy()

    selected_station_ids = read_selected_station_ids(Path(args.selected_stations_file))
    selected_set = set(selected_station_ids)
    control = expanded[expanded["station"].isin(selected_set)].copy()
    control_stations = sorted(control["station"].astype(str).unique().tolist())

    out_expanded = Path(args.output_expanded_csv)
    out_control = Path(args.output_control_csv)
    out_expanded.parent.mkdir(parents=True, exist_ok=True)
    out_control.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_csv(out_expanded, index=False)
    control.to_csv(out_control, index=False)

    out_exp_st_txt = Path(args.output_expanded_stations_txt)
    out_exp_st_csv = Path(args.output_expanded_stations_csv)
    out_exp_st_txt.parent.mkdir(parents=True, exist_ok=True)
    out_exp_st_csv.parent.mkdir(parents=True, exist_ok=True)
    out_exp_st_txt.write_text(
        "\n".join(expanded_stations) + ("\n" if expanded_stations else ""),
        encoding="utf-8",
    )
    pd.DataFrame({"station_id": expanded_stations}).to_csv(out_exp_st_csv, index=False)

    summary = {
        "input_csv": str(src.resolve()),
        "year_policy": [int(args.year_start), int(args.year_end)],
        "min_station_rows": int(args.min_station_rows),
        "input_rows_after_window": int(len(df)),
        "input_station_count_after_window": int(df["station"].nunique()),
        "expanded_rows": int(len(expanded)),
        "expanded_station_count": int(expanded["station"].nunique()),
        "control_rows": int(len(control)),
        "control_station_count": int(control["station"].nunique()),
        "selected_station_count_requested": int(len(selected_station_ids)),
        "selected_station_count_intersection": int(len(control_stations)),
        "expanded_station_ids": expanded_stations,
        "control_station_ids": control_stations,
        "output_paths": {
            "expanded_csv": str(out_expanded.resolve()),
            "control_csv": str(out_control.resolve()),
            "expanded_station_ids_txt": str(out_exp_st_txt.resolve()),
            "expanded_station_ids_csv": str(out_exp_st_csv.resolve()),
        },
    }
    save_json(Path(args.summary_json), summary)

    print(f"Saved expanded CSV: {out_expanded}")
    print(f"Saved control CSV: {out_control}")
    print(f"Saved expanded station list: {out_exp_st_txt} ({len(expanded_stations)} stations)")
    print(f"Saved summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
