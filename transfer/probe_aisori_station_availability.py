#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright


DEFAULT_URL = "http://aisori-m.meteo.ru/aisori-m/index0.xhtml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка наличия station ids в разделах/источниках AISORI")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--username", default=os.getenv("AISORI_USER"))
    parser.add_argument("--password", default=os.getenv("AISORI_PASS"))
    parser.add_argument("--station-ids", default=None, help="Comma-separated station ids")
    parser.add_argument("--station-ids-file", default=None, help="Text file with station ids")
    parser.add_argument(
        "--output-dir",
        default="tmp_rosgidromet_probe/aisori_station_probe",
        help="Directory for probe artifacts",
    )
    parser.add_argument("--headful", action="store_true")
    args = parser.parse_args()
    if not args.username or not args.password:
        raise SystemExit("Username/password are required via args or AISORI_USER/AISORI_PASS.")
    return args


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


def extract_station_ids(page) -> list[str]:
    option_values: list[str] = page.eval_on_selector_all(
        "#form1\\:hlist1_input option",
        "els => els.map(e => e.value)",
    )
    ids: list[str] = []
    for value in option_values:
        m = re.match(r"^\s*(\d{5,6})\b", value or "")
        if m:
            ids.append(m.group(1))
    return sorted(set(ids), key=int)


def wait_options_loaded(page, timeout_sec: float = 12.0) -> int:
    deadline = time.time() + timeout_sec
    last = -1
    stable = 0
    while True:
        count = page.locator("#form1\\:hlist1_input option").count()
        if count == last:
            stable += 1
        else:
            stable = 0
            last = count
        if count > 0 and stable >= 2:
            return count
        if time.time() > deadline:
            return count
        page.wait_for_timeout(250)


def get_select_labels(page, selector: str) -> list[str]:
    options = page.eval_on_selector_all(
        selector + " option",
        "els => els.map(e => ({value: e.value, text: (e.textContent || '').trim()}))",
    )
    labels: list[str] = []
    for row in options:
        value = (row.get("value") or "").strip()
        text = (row.get("text") or "").strip()
        if not value or not text:
            continue
        labels.append(text)
    return labels


def get_select_options(page, selector: str) -> list[dict[str, str]]:
    options = page.eval_on_selector_all(
        selector + " option",
        "els => els.map(e => ({value: (e.value || '').trim(), text: ((e.textContent || '').trim())}))",
    )
    out: list[dict[str, str]] = []
    for row in options:
        value = (row.get("value") or "").strip()
        text = (row.get("text") or "").strip()
        if not value:
            continue
        out.append({"value": value, "text": text})
    return out


def main() -> int:
    args = parse_args()
    requested = parse_station_ids(args.station_ids, args.station_ids_file)
    if not requested:
        raise SystemExit("No requested station ids. Use --station-ids or --station-ids-file.")

    outdir = Path(args.output_dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=not args.headful)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=120_000)
        page.fill("#j_idt13\\:usr", args.username)
        page.fill("#j_idt13\\:pwd", args.password)
        page.click("#j_idt13\\:j_idt27")
        page.wait_for_selector("#form1\\:newbut1", timeout=120_000)
        page.click("#form1\\:newbut1")
        page.wait_for_selector("#form1\\:razbd", timeout=120_000)
        page.wait_for_selector("#form1\\:istd", timeout=120_000)

        sections = get_select_options(page, "#form1\\:razbd")
        for section in sections:
            page.select_option("#form1\\:razbd", value=section["value"])
            wait_options_loaded(page)
            datasets = get_select_options(page, "#form1\\:istd")
            for dataset in datasets:
                page.select_option("#form1\\:istd", value=dataset["value"])
                wait_options_loaded(page)
                available_ids = extract_station_ids(page)
                inter = [sid for sid in requested if sid in set(available_ids)]
                missing = [sid for sid in requested if sid not in set(available_ids)]
                row = {
                    "section": section["text"],
                    "dataset": dataset["text"],
                    "available_station_count": len(available_ids),
                    "requested_count": len(requested),
                    "intersection_count": len(inter),
                    "intersection_ids": ",".join(inter),
                    "missing_count": len(missing),
                }
                rows.append(row)
                details.append(
                    {
                        **row,
                        "requested_ids": requested,
                        "intersection_ids_list": inter,
                        "missing_ids": missing,
                        "available_ids_sample_head": available_ids[:25],
                        "available_ids_sample_tail": available_ids[-25:],
                    }
                )

        page.screenshot(path=str(outdir / "selection_page_after_probe.png"), full_page=True)
        ctx.close()
        browser.close()

    df = pd.DataFrame(rows).sort_values(["intersection_count", "available_station_count"], ascending=[False, False])
    df.to_csv(outdir / "probe_summary.csv", index=False)
    (outdir / "probe_summary.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    best = df.head(1).to_dict("records")
    payload = {
        "requested_station_ids": requested,
        "rows": int(len(df)),
        "best": best[0] if best else None,
        "summary_csv": str((outdir / "probe_summary.csv").resolve()),
        "summary_json": str((outdir / "probe_summary.json").resolve()),
    }
    (outdir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
