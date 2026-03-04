from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

RAG_RUNTIME_API_DIR = Path(__file__).resolve().parents[2] / "API"
if str(RAG_RUNTIME_API_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_RUNTIME_API_DIR))

import data_quality_checks as dq


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def test_parse_date_supports_offset_without_colon() -> None:
    parsed = dq.parse_date("2026-03-04T10:30:00+0000")
    assert parsed == datetime(2026, 3, 4, 10, 30, tzinfo=timezone.utc)


def test_parse_date_sets_utc_for_naive_datetime() -> None:
    parsed = dq.parse_date("2026-03-04T10:30:00")
    assert parsed.tzinfo == timezone.utc
    assert parsed.hour == 10
    assert parsed.minute == 30


def test_check_required_fields_passes_for_valid_records() -> None:
    events = [
        {"event_id": "1", "date_start": _iso(datetime.now(timezone.utc))},
        {"event_id": "2", "date_start": _iso(datetime.now(timezone.utc))},
    ]
    dq.check_required_fields(events)


def test_check_required_fields_raises_for_missing_date_start() -> None:
    events = [{"event_id": "1"}]
    with pytest.raises(AssertionError, match="events without required fields"):
        dq.check_required_fields(events)


def test_check_date_window_raises_for_too_old_event() -> None:
    too_old = datetime.now(timezone.utc) - timedelta(days=500)
    events = [{"event_id": "old", "date_start": _iso(too_old)}]

    with pytest.raises(AssertionError, match="events older than"):
        dq.check_date_window(events, max_days_old=375)


def test_check_normandie_geo_accepts_known_departments() -> None:
    events = [
        {"event_id": "1", "department": "calvados"},
        {"event_id": "2", "department": "Seine-Maritime"},
        {"event_id": "3", "location_department": "normandie"},
    ]
    dq.check_normandie_geo(events)


def test_check_normandie_geo_raises_for_outside_region() -> None:
    events = [{"event_id": "x", "department": "ile-de-france"}]

    with pytest.raises(AssertionError, match="events outside Normandie"):
        dq.check_normandie_geo(events)
