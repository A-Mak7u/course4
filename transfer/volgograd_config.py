from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "volgograd"
RAW_ROOT = DATA_ROOT / "raw"
INTERIM_ROOT = DATA_ROOT / "interim"
PROCESSED_ROOT = DATA_ROOT / "processed"


@dataclass(frozen=True)
class RegionSpec:
    name: str
    start_year: int
    end_year: int
    south: float
    west: float
    north: float
    east: float
    country: str
    region_codes: tuple[str, ...]


VOLGOGRAD_SPEC = RegionSpec(
    name="volgograd",
    start_year=2013,
    end_year=2023,
    south=47.0,
    west=40.0,
    north=52.2,
    east=47.5,
    country="RU",
    region_codes=("VGG", "VG"),
)


def ensure_region_dirs() -> None:
    for path in (DATA_ROOT, RAW_ROOT, INTERIM_ROOT, PROCESSED_ROOT):
        path.mkdir(parents=True, exist_ok=True)
