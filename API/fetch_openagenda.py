import argparse
import json
import time

import pandas as pd
import requests

try:
    from .paths import RAW_CSV_PATH, RAW_JSON_PATH, ensure_directories
except ImportError:
    from paths import RAW_CSV_PATH, RAW_JSON_PATH, ensure_directories


OPEN_DATASETS_API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)
OPEN_DATASETS_EXPORT_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/exports/csv"
)


def fetch_records(region: str = "Normandie", limit: int = 100, sleep_s: float = 0.2) -> list[dict]:
    """Fetch records with pagination from OpenDataSoft."""
    offset = 0
    all_records: list[dict] = []

    while True:
        params = {
            "limit": limit,
            "offset": offset,
            "where": f'location_region="{region}"',
        }
        response = requests.get(OPEN_DATASETS_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        records = data.get("results", [])
        if not records:
            break

        all_records.extend(records)
        offset += limit
        print(f"{len(all_records)} records fetched...")
        time.sleep(sleep_s)

    return all_records


def save_records(records: list[dict], json_path=RAW_JSON_PATH, csv_path=RAW_CSV_PATH) -> None:
    ensure_directories()
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


def download_csv_export(region: str = "Normandie", output_path=RAW_CSV_PATH) -> None:
    """Download full CSV export directly from OpenDataSoft."""
    ensure_directories()
    params = {"where": f'location_region="{region}"'}
    with requests.get(OPEN_DATASETS_EXPORT_URL, params=params, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(output_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    print(f"Export saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OpenAgenda data for Normandie.")
    parser.add_argument(
        "--mode",
        choices=["paginated", "export"],
        default="paginated",
        help="paginated: records API; export: CSV export endpoint",
    )
    parser.add_argument("--region", default="Normandie")
    args = parser.parse_args()

    if args.mode == "paginated":
        records = fetch_records(region=args.region)
        save_records(records)
        print(f"Done. Total records: {len(records)}")
    else:
        download_csv_export(region=args.region)


if __name__ == "__main__":
    main()
