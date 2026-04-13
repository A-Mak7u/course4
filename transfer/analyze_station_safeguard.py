from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Анализ safeguard-эффекта: xgb_delta_global vs xgb_delta_gated по станциям"
    )
    parser.add_argument("--risk-details-csv", required=True)
    parser.add_argument("--global-variant", default="xgb_delta_global")
    parser.add_argument("--gated-variant", default="xgb_delta_gated")
    parser.add_argument(
        "--output-station-csv",
        default=None,
        help="Итоговая station-таблица. По умолчанию рядом с risk-details.",
    )
    parser.add_argument(
        "--output-summary-json",
        default=None,
        help="JSON-сводка safeguard. По умолчанию рядом с risk-details.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.risk_details_csv)
    if not src.exists():
        raise FileNotFoundError(f"Risk details CSV not found: {src}")

    df = pd.read_csv(src).copy()
    required = {"variant", "station", "mae_gain"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Risk details CSV missing columns: {sorted(missing)}")

    global_df = (
        df[df["variant"] == args.global_variant][["station", "mae_gain"]]
        .rename(columns={"mae_gain": "mae_gain_global"})
        .copy()
    )
    gated_df = (
        df[df["variant"] == args.gated_variant][["station", "mae_gain"]]
        .rename(columns={"mae_gain": "mae_gain_gated"})
        .copy()
    )
    joined = global_df.merge(gated_df, on="station", how="inner")
    if joined.empty:
        raise RuntimeError("No overlapping station rows for global/gated variants")

    joined["global_worse_than_baseline"] = joined["mae_gain_global"] < 0
    joined["gated_worse_than_baseline"] = joined["mae_gain_gated"] < 0
    joined["safeguard_delta"] = joined["mae_gain_gated"] - joined["mae_gain_global"]
    joined["safeguard_recovered"] = joined["global_worse_than_baseline"] & (~joined["gated_worse_than_baseline"])
    joined["still_worse_after_gated"] = joined["global_worse_than_baseline"] & joined["gated_worse_than_baseline"]
    joined = joined.sort_values(["global_worse_than_baseline", "safeguard_delta"], ascending=[False, True]).reset_index(drop=True)

    heavy = joined[joined["global_worse_than_baseline"]].copy()
    summary = {
        "risk_details_csv": str(src.resolve()),
        "global_variant": args.global_variant,
        "gated_variant": args.gated_variant,
        "stations_total": int(len(joined)),
        "heavy_station_count_global": int(len(heavy)),
        "recovered_by_gated_count": int(heavy["safeguard_recovered"].sum()) if len(heavy) else 0,
        "still_worse_after_gated_count": int(heavy["still_worse_after_gated"].sum()) if len(heavy) else 0,
        "global_worse_count": int((joined["mae_gain_global"] < 0).sum()),
        "gated_worse_count": int((joined["mae_gain_gated"] < 0).sum()),
        "mean_safeguard_delta_all": float(joined["safeguard_delta"].mean()),
        "mean_safeguard_delta_heavy_only": float(heavy["safeguard_delta"].mean()) if len(heavy) else 0.0,
    }

    if args.output_station_csv:
        out_station = Path(args.output_station_csv)
    else:
        out_station = src.parent / "safeguard_station_analysis.csv"
    if args.output_summary_json:
        out_summary = Path(args.output_summary_json)
    else:
        out_summary = src.parent / "safeguard_summary.json"

    joined.to_csv(out_station, index=False)
    save_json(out_summary, summary)
    print(f"Saved safeguard station analysis: {out_station}")
    print(f"Saved safeguard summary: {out_summary}")
    print(f"Heavy stations (global worsened): {summary['heavy_station_count_global']}")
    print(f"Recovered by gated: {summary['recovered_by_gated_count']}")
    print(f"Still worsened after gated: {summary['still_worse_after_gated_count']}")


if __name__ == "__main__":
    main()
