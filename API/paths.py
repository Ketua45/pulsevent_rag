from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "data_raw"
PROCESSED_DIR = DATA_DIR / "data_processed"

RAW_CSV_PATH = RAW_DIR / "evenements_normandie.csv"
RAW_JSON_PATH = RAW_DIR / "evenements_normandie.json"
WINDOW_JSON_PATH = PROCESSED_DIR / "normandie_1y_data.json"
PROCESSED_JSONL_PATH = PROCESSED_DIR / "events_processed.jsonl"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore_normandie"


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

