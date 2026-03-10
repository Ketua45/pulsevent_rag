import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from .paths import PROCESSED_JSONL_PATH
except ImportError:
    from paths import PROCESSED_JSONL_PATH


NORMANDIE_DEPARTMENTS = {
    "calvados",
    "eure",
    "manche",
    "orne",
    "seine-maritime",
    "seine maritime",
    "normandie",
    "manche et orne",
}


def load_events(path=PROCESSED_JSONL_PATH) -> list[dict]:
    assert path.exists(), f"File not found: {path}"
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    print(f"{len(events)} events loaded")
    return events


def parse_date(date_str):
    if not date_str:
        raise ValueError("empty date")

    text = date_str.strip()
    if len(text) >= 5 and text[-5] in ["+", "-"] and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]

    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def check_required_fields(events):
    required_fields = {"event_id", "date_start"}
    missing = []
    for record in events:
        if not required_fields.issubset(set(record.keys())):
            missing.append(record.get("event_id"))

    assert len(missing) == 0, f"{len(missing)} events without required fields"
    print("Check required_fields: OK")


def check_date_window(events, max_days_old: int = 385):
    now = datetime.now(timezone.utc)
    min_date = now - timedelta(days=max_days_old)

    too_old = []
    for record in events:
        try:
            dt = parse_date(record["date_start"])
            if dt < min_date:
                too_old.append((record["event_id"], dt.isoformat()))
        except Exception as exc:
            too_old.append((record.get("event_id"), str(exc)))

    assert len(too_old) == 0, f"{len(too_old)} events older than {max_days_old} days"
    print("Check date_window: OK")


def check_normandie_geo(events):
    bad_geo = []
    for record in events:
        dep = record.get("department") or record.get("location_department") or ""
        dep_norm = str(dep).strip().lower()
        if dep_norm not in NORMANDIE_DEPARTMENTS:
            bad_geo.append((record.get("event_id"), dep))

    assert len(bad_geo) == 0, f"{len(bad_geo)} events outside Normandie"
    print("Check normandie_geo: OK")


def run_all_checks(path=PROCESSED_JSONL_PATH):
    events = load_events(path)
    check_required_fields(events)
    check_date_window(events)
    check_normandie_geo(events)
    print("All checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data quality checks on events_processed.jsonl")
    parser.add_argument("--path", default=str(PROCESSED_JSONL_PATH))
    args = parser.parse_args()
    run_all_checks(path=Path(args.path))


if __name__ == "__main__":
    main()
