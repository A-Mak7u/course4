#!/usr/bin/env python3
"""Probe the public Rosgidromet EIP OData catalog without authentication.

This script focuses on what is publicly available from https://eip.meteo.ru/api:
- product metadata
- service metadata
- service territory catalog
- subject and department dictionaries
- API metadata entity sets

Outputs are written to an ignored tmp_* directory by default so the probe can be
rerun freely without polluting git.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import urllib3


DEFAULT_PRODUCT_IDS = [51, 54, 55, 56, 157, 159, 163]
DEFAULT_OUTPUT_DIR = "tmp_rosgidromet_probe/latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="https://eip.meteo.ru/api",
        help="Public EIP OData API base URL.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for raw dumps and derived summaries.",
    )
    parser.add_argument(
        "--product-ids",
        nargs="+",
        type=int,
        default=DEFAULT_PRODUCT_IDS,
        help="Catalog product IDs to probe.",
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        help="Enable TLS certificate verification. Disabled by default because "
        "eip.meteo.ru currently presents an incomplete certificate chain.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class ProbeConfig:
    base_url: str
    verify_ssl: bool


class EipClient:
    def __init__(self, config: ProbeConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.verify_ssl = config.verify_ssl
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "course4-rosgidromet-probe/1.0",
                "Accept": "application/json, text/plain, */*",
            }
        )
        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=60, verify=self.verify_ssl)
        response.raise_for_status()
        return response.json()

    def get_text(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, params=params, timeout=60, verify=self.verify_ssl)
        response.raise_for_status()
        return response.text


def extract_entity_sets(metadata_xml: str) -> list[str]:
    return re.findall(r'EntitySet Name="([^"]+)"', metadata_xml)


def extract_entity_types(metadata_xml: str) -> list[str]:
    return re.findall(r'EntityType Name="([^"]+)"', metadata_xml)


def summarize_services(
    services_by_product: dict[int, list[dict[str, Any]]],
    territories_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for product_id, services in services_by_product.items():
        for service in services:
            territory_links = service.get("Territories") or []
            territory_ids = [link.get("ServiceTerritoryId") for link in territory_links if link.get("ServiceTerritoryId")]
            territory_objects = [territories_by_id[tid] for tid in territory_ids if tid in territories_by_id]

            subject_titles = sorted(
                {
                    ((territory.get("Subject") or {}).get("Title"))
                    for territory in territory_objects
                    if (territory.get("Subject") or {}).get("Title")
                }
            )
            territory_types = sorted({territory.get("TerritoryType") for territory in territory_objects if territory.get("TerritoryType")})

            rows.append(
                {
                    "product_id": product_id,
                    "product_title": (service.get("Product") or {}).get("Title"),
                    "service_id": service.get("Id"),
                    "service_title": service.get("Title"),
                    "department_id": service.get("DepartmentId"),
                    "department_title": (service.get("Department") or {}).get("Title"),
                    "request_processing_strategy": service.get("RequestProcessingStrategy"),
                    "provision_method": service.get("ProvisionMethod"),
                    "metadata": service.get("Metadata"),
                    "export": service.get("Export"),
                    "online_payment": service.get("OnlinePaymentIsAvailable"),
                    "provision_form": service.get("ProvisionForm"),
                    "provision_term_info": service.get("ProvisionTermInfo"),
                    "territory_link_count": len(territory_ids),
                    "resolved_territory_count": len(territory_objects),
                    "subject_count": len(subject_titles),
                    "subject_titles": "; ".join(subject_titles),
                    "territory_types": "; ".join(territory_types),
                }
            )
    return rows


def build_findings(
    entity_sets: list[str],
    service_rows: list[dict[str, Any]],
    top_probe_error: str | None,
) -> dict[str, Any]:
    joined = "\n".join(entity_sets).lower()
    return {
        "catalog_api_public": True,
        "order_like_entities_present": any(
            token in joined for token in ["order", "request", "basket", "cart", "ticket"]
        ),
        "catalog_entity_sets": entity_sets,
        "top_query_allowed": top_probe_error is None,
        "top_query_error": top_probe_error,
        "target_services_all_require_personal_delivery": all(
            "кабинет" in (row.get("provision_method") or "").lower()
            or "почт" in (row.get("provision_method") or "").lower()
            or "электрон" in (row.get("provision_method") or "").lower()
            for row in service_rows
        ),
        "target_services_summary": [
            {
                "service_id": row["service_id"],
                "product_id": row["product_id"],
                "service_title": row["service_title"],
                "provision_method": row["provision_method"],
                "request_processing_strategy": row["request_processing_strategy"],
                "metadata": row["metadata"],
                "department_title": row["department_title"],
            }
            for row in service_rows
        ],
    }


def render_summary_markdown(
    product_ids: list[int],
    product_map: dict[int, dict[str, Any]],
    service_rows: list[dict[str, Any]],
    findings: dict[str, Any],
) -> str:
    lines = [
        "# Rosgidromet Public API Probe",
        "",
        f"- Base URL: `{findings.get('catalog_entity_sets') and 'https://eip.meteo.ru/api'}`",
        f"- Target product IDs: `{', '.join(map(str, product_ids))}`",
        f"- Public catalog API available: `{findings['catalog_api_public']}`",
        f"- Order/request entities present in `$metadata`: `{findings['order_like_entities_present']}`",
        f"- `$top` accepted by API: `{findings['top_query_allowed']}`",
    ]
    if findings.get("top_query_error"):
        lines.append(f"- `$top` probe error: `{findings['top_query_error']}`")

    lines.extend(["", "## Products", ""])
    for product_id in product_ids:
        product = product_map[product_id]
        lines.append(
            f"- `{product_id}` `{product.get('Title')}` | code `{product.get('Code')}` | type `{product.get('TypeId')}`"
        )

    lines.extend(["", "## Services", ""])
    for row in service_rows:
        lines.append(
            "- "
            f"product `{row['product_id']}` -> service `{row['service_id']}` "
            f"`{row['service_title']}` | method `{row['provision_method']}` | "
            f"strategy `{row['request_processing_strategy']}` | metadata `{row['metadata']}` | "
            f"department `{row['department_title']}` | territories `{row['resolved_territory_count']}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Public `/api` exposes the product/service catalog and supporting dictionaries.",
            "- The public OData metadata does not expose order/request/cart/ticket entities.",
            "- For the probed station-data products, delivery metadata points to `Личный кабинет` or manual fulfillment rather than anonymous file endpoints.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    client = EipClient(ProbeConfig(base_url=args.base_url, verify_ssl=args.verify_ssl))

    metadata_xml = client.get_text("$metadata")
    entity_sets = extract_entity_sets(metadata_xml)
    entity_types = extract_entity_types(metadata_xml)

    products: list[dict[str, Any]] = []
    services_by_product: dict[int, list[dict[str, Any]]] = {}
    for product_id in args.product_ids:
        product = client.get_json(f"product({product_id})")
        products.append(product)
        services_by_product[product_id] = client.get_json(
            "service",
            params={
                "$filter": f"ProductId eq {product_id}",
                "$expand": "Department,Product,Territories",
            },
        ).get("value", [])

    subjects = client.get_json(
        "subject",
        params={
            "$orderby": "Title",
            "$expand": "Okrug",
        },
    ).get("value", [])
    departments = client.get_json(
        "department",
        params={
            "$select": "Id,Title",
            "$orderby": "Title",
        },
    ).get("value", [])
    service_territories = client.get_json(
        "serviceterritory",
        params={
            "$expand": "Subject,Sea,Ocean,Lake,Okrug,City,CustomTerritory",
        },
    ).get("value", [])

    top_probe_error = None
    try:
        client.get_json("subject", params={"$top": "1"})
    except requests.HTTPError as exc:
        response = exc.response
        if response is not None:
            try:
                payload = response.json()
                top_probe_error = payload.get("error", {}).get("message")
            except json.JSONDecodeError:
                top_probe_error = response.text[:500]

    territories_by_id = {row["Id"]: row for row in service_territories if row.get("Id") is not None}
    product_map = {row["Id"]: row for row in products}
    service_rows = summarize_services(services_by_product, territories_by_id)
    findings = build_findings(entity_sets, service_rows, top_probe_error)

    relevant_territory_ids = sorted(
        {
            link["ServiceTerritoryId"]
            for services in services_by_product.values()
            for service in services
            for link in (service.get("Territories") or [])
            if link.get("ServiceTerritoryId") is not None
        }
    )
    relevant_territories = [territories_by_id[tid] for tid in relevant_territory_ids if tid in territories_by_id]

    write_text(output_dir / "api_metadata.xml", metadata_xml)
    write_json(output_dir / "api_entities.json", {"entity_sets": entity_sets, "entity_types": entity_types})
    write_json(output_dir / "products.json", products)
    write_json(output_dir / "services_by_product.json", services_by_product)
    write_json(output_dir / "subjects.json", subjects)
    write_json(output_dir / "departments.json", departments)
    write_json(output_dir / "relevant_service_territories.json", relevant_territories)
    write_json(output_dir / "findings.json", findings)
    write_csv(
        output_dir / "service_summary.csv",
        service_rows,
        fieldnames=[
            "product_id",
            "product_title",
            "service_id",
            "service_title",
            "department_id",
            "department_title",
            "request_processing_strategy",
            "provision_method",
            "metadata",
            "export",
            "online_payment",
            "provision_form",
            "provision_term_info",
            "territory_link_count",
            "resolved_territory_count",
            "subject_count",
            "subject_titles",
            "territory_types",
        ],
    )
    write_text(
        output_dir / "SUMMARY.md",
        render_summary_markdown(args.product_ids, product_map, service_rows, findings),
    )

    print(f"Saved probe outputs to: {output_dir}")
    print(f"Products: {len(products)} | Services: {len(service_rows)} | Subjects: {len(subjects)}")
    if top_probe_error:
        print(f"Top query error: {top_probe_error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
