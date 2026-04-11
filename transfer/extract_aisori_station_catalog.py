from __future__ import annotations

import argparse
import glob
import re
import zipfile
from pathlib import Path

import pandas as pd

from pipeline_common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Извлечение каталога станций из AISORI statlist*.txt")
    parser.add_argument("--zip-glob", default="data/rosgidromet/aisori/wr*.zip")
    parser.add_argument("--encoding", default="cp1251")
    parser.add_argument(
        "--output-csv",
        default="data/rosgidromet/aisori/aisori_station_catalog.csv",
        help="Итоговый CSV каталога станций.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="JSON со сводкой. По умолчанию: <output-csv>.summary.json",
    )
    return parser.parse_args()


def parse_station_line(line: str) -> tuple[str, str, str] | None:
    line = line.strip()
    if not line:
        return None
    m = re.match(r"^(\d{5,6})\s+(.+)$", line)
    if not m:
        return None
    station = m.group(1)
    rest = m.group(2).strip()
    parts = [p.strip() for p in re.split(r"\s{2,}", rest) if p.strip()]
    if len(parts) >= 2:
        name = " ".join(parts[:-1]).strip()
        country = parts[-1].strip()
    else:
        name = rest
        country = ""
    return station, name, country


def main() -> None:
    args = parse_args()
    zip_paths = sorted(glob.glob(args.zip_glob))
    if not zip_paths:
        raise RuntimeError(f"Не найдены AISORI zip по паттерну: {args.zip_glob}")

    rows: list[dict[str, str]] = []
    for zpath in zip_paths:
        with zipfile.ZipFile(zpath, "r") as zf:
            statlist_entries = [name for name in zf.namelist() if "statlist" in Path(name).name.lower()]
            if not statlist_entries:
                continue
            entry = statlist_entries[0]
            content = zf.read(entry).decode(args.encoding, errors="ignore")
            for raw_line in content.splitlines():
                parsed = parse_station_line(raw_line)
                if not parsed:
                    continue
                station, name, country = parsed
                rows.append(
                    {
                        "station": station,
                        "name": name,
                        "country": country,
                        "source_zip": Path(zpath).name,
                        "source_entry": Path(entry).name,
                    }
                )

    if not rows:
        raise RuntimeError("Не удалось извлечь ни одной станции из statlist*.txt")

    raw = pd.DataFrame(rows)
    raw_unique = raw.drop_duplicates().reset_index(drop=True)
    grouped = raw_unique.groupby("station", as_index=False)
    catalog = grouped.agg(
        name=("name", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        country=("country", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
        name_variants=("name", "nunique"),
        archives_seen=("source_zip", "nunique"),
    )
    catalog = catalog.sort_values("station", key=lambda s: s.astype(int)).reset_index(drop=True)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_csv, index=False)

    summary_path = Path(args.summary_json) if args.summary_json else out_csv.with_suffix(out_csv.suffix + ".summary.json")
    summary = {
        "zip_glob": args.zip_glob,
        "zip_files_used": [str(Path(p).resolve()) for p in zip_paths],
        "encoding": args.encoding,
        "rows_raw": int(len(raw)),
        "rows_unique": int(len(raw_unique)),
        "stations_total": int(catalog["station"].nunique()),
        "output_csv": str(out_csv.resolve()),
        "station_min": str(catalog["station"].min()) if not catalog.empty else None,
        "station_max": str(catalog["station"].max()) if not catalog.empty else None,
    }
    save_json(summary_path, summary)

    print(f"Saved station catalog: {out_csv}")
    print(f"Saved summary: {summary_path}")
    print(f"Stations: {summary['stations_total']}")


if __name__ == "__main__":
    main()
