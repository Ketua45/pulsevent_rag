import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    from .paths import (
        PROCESSED_JSONL_PATH,
        RAW_CSV_PATH,
        WINDOW_JSON_PATH,
        ensure_directories,
    )
except ImportError:
    from paths import (
        PROCESSED_JSONL_PATH,
        RAW_CSV_PATH,
        WINDOW_JSON_PATH,
        ensure_directories,
    )


DEFAULT_COLUMNS_TO_DROP = [
    "location_insee",
    "location_countrycode",
    "country_fr",
    "location_uid",
    "location_region",
    "updatedat",
    "daterange_fr",
    "firstdate_end",
    "lastdate_begin",
    "timings",
    "originagenda_uid",
    "originagenda_title",
    "status",
]


def safe_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return ""
    return str(value).strip()


def build_document_text(row: pd.Series) -> str:
    title = safe_text(row.get("title_fr"))
    short = safe_text(row.get("description_fr"))
    longd = safe_text(row.get("longdescription_fr"))
    description = longd if longd else short
    if not description:
        description = title

    return (
        f"Titre: {title}\n"
        f"Description: {description}\n"
        f"Date: {row.get('date_start')} -> {row.get('date_end')}\n"
        f"Lieu: {safe_text(row.get('location_name'))}, {safe_text(row.get('location_address'))}, "
        f"{safe_text(row.get('location_postalcode'))} {safe_text(row.get('location_city'))} "
        f"({safe_text(row.get('location_department'))})\n"
        f"Slug: {safe_text(row.get('slug'))}\n"
    )


def preprocess_raw_csv_to_window_json(
    raw_csv_path: Path = RAW_CSV_PATH,
    output_json_path: Path = WINDOW_JSON_PATH,
    days_past: int = 365,
    days_future: int = 365,
) -> pd.DataFrame:
    """From raw CSV to a first filtered JSON file."""
    ensure_directories()
    df = pd.read_csv(raw_csv_path, encoding="utf-8", sep=";")

    # Remove very sparse columns, URL/image columns, and known noisy fields.
    missing_percentages = df.isnull().mean() * 100
    sparse_cols = missing_percentages[missing_percentages > 50].index.tolist()
    if sparse_cols:
        df = df.drop(columns=sparse_cols, errors="ignore")

    url_cols = [c for c in df.columns if "url" in c.lower() or "image" in c.lower() or "thumbnail" in c.lower()]
    if url_cols:
        df = df.drop(columns=url_cols, errors="ignore")

    df = df.drop(columns=DEFAULT_COLUMNS_TO_DROP, errors="ignore")

    df["firstdate_begin"] = pd.to_datetime(df["firstdate_begin"], errors="coerce")
    if df["firstdate_begin"].dt.tz is not None:
        df["firstdate_begin"] = df["firstdate_begin"].dt.tz_localize(None)

    now = pd.Timestamp.now()
    min_date = now - pd.DateOffset(days=days_past)
    max_date = now + pd.DateOffset(days=days_future)
    filtered = df[
        df["firstdate_begin"].notna()
        & (df["firstdate_begin"] >= min_date)
        & (df["firstdate_begin"] <= max_date)
    ].copy()

    filtered.to_json(output_json_path)
    print(f"Saved filtered window JSON: {output_json_path} | rows={len(filtered)}")
    return filtered


def preprocess_window_json_to_events_jsonl(
    window_json_path: Path = WINDOW_JSON_PATH,
    output_jsonl_path: Path = PROCESSED_JSONL_PATH,
    days_past: int = 365,
    days_future: int = 365,
) -> pd.DataFrame:
    """From window JSON to final events_processed.jsonl for vectorization."""
    ensure_directories()
    data = json.load(open(window_json_path, "r", encoding="utf-8"))
    df = pd.DataFrame(data)

    # Datetime and coordinates.
    df["date_start"] = pd.to_datetime(df["firstdate_begin"], unit="ms", utc=True, errors="coerce")
    df["date_end"] = pd.to_datetime(df["lastdate_end"], utc=True, errors="coerce")

    coords = df["location_coordinates"].astype(str).str.split(",", n=1, expand=True)
    df["lat"] = pd.to_numeric(coords[0], errors="coerce")
    df["lon"] = pd.to_numeric(coords[1], errors="coerce")

    now = datetime.now(timezone.utc)
    start_min = now - timedelta(days=days_past)
    start_max = now + timedelta(days=days_future)
    df = df[df["date_start"].between(start_min, start_max)].copy()

    # Text cleanup + title fallback.
    df["title_fr"] = df["title_fr"].apply(safe_text)
    df["slug"] = df["slug"].apply(safe_text)
    df["uid"] = df["uid"].astype(str)

    mask_empty_title = df["title_fr"].eq("")
    df.loc[mask_empty_title, "title_fr"] = df.loc[mask_empty_title, "slug"]
    mask_empty_title2 = df["title_fr"].eq("")
    df.loc[mask_empty_title2, "title_fr"] = df.loc[mask_empty_title2, "uid"]

    df["document_text"] = df.apply(build_document_text, axis=1)

    processed = pd.DataFrame(
        {
            "event_id": df["uid"].astype(str),
            "slug": df["slug"].apply(safe_text),
            "title": df["title_fr"].apply(safe_text),
            "description": df["description_fr"].apply(safe_text),
            "long_description": df["longdescription_fr"].apply(safe_text),
            "conditions": df["conditions_fr"].apply(safe_text),
            "date_start": df["date_start"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "date_end": df["date_end"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "location_name": df["location_name"].apply(safe_text),
            "address": df["location_address"].apply(safe_text),
            "postal_code": df["location_postalcode"].apply(safe_text),
            "city": df["location_city"].apply(safe_text),
            "department": df["location_department"].apply(safe_text),
            "lat": df["lat"],
            "lon": df["lon"],
            "document_text": df["document_text"],
            "source": "openagenda",
        }
    )

    assert processed["event_id"].notna().all()
    assert processed["event_id"].is_unique
    assert processed["title"].astype(str).str.strip().ne("").all()
    assert processed["date_start"].notna().all()

    processed.to_json(output_jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved processed JSONL: {output_jsonl_path} | rows={len(processed)}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess OpenAgenda events for RAG.")
    parser.add_argument(
        "--stage",
        choices=["window", "processed", "all"],
        default="all",
        help="window: raw CSV -> normandie_1y_data.json, processed: window JSON -> events_processed.jsonl",
    )
    args = parser.parse_args()

    if args.stage in {"window", "all"}:
        preprocess_raw_csv_to_window_json()
    if args.stage in {"processed", "all"}:
        preprocess_window_json_to_events_jsonl()


if __name__ == "__main__":
    main()
