#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe EIP API for target station codes and service channels")
    parser.add_argument("--base-url", default="https://eip.meteo.ru/api")
    parser.add_argument("--station-ids", default=None, help="Comma-separated station ids")
    parser.add_argument("--station-ids-file", default=None, help="Text file with station ids")
    parser.add_argument(
        "--product-ids",
        default="54,55,56",
        help="Comma-separated product ids for service probe (default: 54,55,56)",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp_rosgidromet_probe/eip_station_probe",
        help="Directory for probe artifacts",
    )
    return parser.parse_args()


def parse_station_ids(raw: str | None, file_path: str | None) -> list[str]:
    ids: list[str] = []
    if raw:
        ids.extend([x.strip() for x in raw.split(",") if x.strip()])
    if file_path:
        text = Path(file_path).read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split()[0])

    out: list[str] = []
    seen: set[str] = set()
    for sid in ids:
        sid = "".join(ch for ch in sid if ch.isdigit())
        if not sid:
            continue
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def parse_product_ids(raw: str) -> list[int]:
    out: list[int] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def http_get_json(base_url: str, path: str, params: dict[str, str] | None = None) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "course4-eip-probe/1.0"})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    return json.loads(payload)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    station_ids = parse_station_ids(args.station_ids, args.station_ids_file)
    if not station_ids:
        raise SystemExit("No station ids. Use --station-ids or --station-ids-file")
    product_ids = parse_product_ids(args.product_ids)

    outdir = Path(args.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    station_rows: list[dict[str, Any]] = []
    station_details: list[dict[str, Any]] = []
    for sid in station_ids:
        data = http_get_json(
            args.base_url,
            "meteostation",
            params={"$filter": f"Code eq '{sid}'"},
        )
        values = data.get("value", []) if isinstance(data, dict) else []
        if values:
            item = values[0]
            station_rows.append(
                {
                    "station_id": sid,
                    "found": 1,
                    "title": item.get("Title"),
                    "subject_id": item.get("SubjectID"),
                    "latitude": item.get("Latitude"),
                    "longitude": item.get("Longitude"),
                    "sid": item.get("Sid"),
                }
            )
            station_details.append({"station_id": sid, "found": True, "items": values})
        else:
            station_rows.append(
                {
                    "station_id": sid,
                    "found": 0,
                    "title": None,
                    "subject_id": None,
                    "latitude": None,
                    "longitude": None,
                    "sid": None,
                }
            )
            station_details.append({"station_id": sid, "found": False, "items": []})

    service_rows: list[dict[str, Any]] = []
    for pid in product_ids:
        data = http_get_json(
            args.base_url,
            "service",
            params={"$filter": f"ProductId eq {pid}", "$expand": "Department,Product"},
        )
        values = data.get("value", []) if isinstance(data, dict) else []
        for item in values:
            service_rows.append(
                {
                    "product_id": pid,
                    "service_id": item.get("Id"),
                    "title": item.get("Title"),
                    "metadata": item.get("Metadata"),
                    "request_processing_strategy": item.get("RequestProcessingStrategy"),
                    "provision_method": item.get("ProvisionMethod"),
                    "department": (item.get("Department") or {}).get("Title"),
                    "export": item.get("Export"),
                }
            )

    station_csv = outdir / "station_codes_lookup.csv"
    service_csv = outdir / "service_channels.csv"
    write_csv(
        station_csv,
        station_rows,
        ["station_id", "found", "title", "subject_id", "latitude", "longitude", "sid"],
    )
    write_csv(
        service_csv,
        service_rows,
        [
            "product_id",
            "service_id",
            "title",
            "metadata",
            "request_processing_strategy",
            "provision_method",
            "department",
            "export",
        ],
    )

    found_ids = [row["station_id"] for row in station_rows if row["found"] == 1]
    missing_ids = [row["station_id"] for row in station_rows if row["found"] == 0]
    automatic_rows = [row for row in service_rows if str(row.get("request_processing_strategy") or "").lower() == "automatic"]
    automatic_with_metadata = [row for row in automatic_rows if row.get("metadata")]

    result = {
        "base_url": args.base_url,
        "requested_station_ids": station_ids,
        "found_count": len(found_ids),
        "missing_count": len(missing_ids),
        "found_station_ids": found_ids,
        "missing_station_ids": missing_ids,
        "product_ids_probed": product_ids,
        "service_rows_count": len(service_rows),
        "automatic_service_count": len(automatic_rows),
        "automatic_with_metadata_count": len(automatic_with_metadata),
        "automatic_with_metadata": automatic_with_metadata,
        "station_csv": str(station_csv.resolve()),
        "service_csv": str(service_csv.resolve()),
    }
    write_json(outdir / "result.json", result)
    write_json(outdir / "station_details.json", station_details)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
