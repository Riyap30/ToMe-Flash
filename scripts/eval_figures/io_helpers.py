import json
import sys
from pathlib import Path


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: missing required file: {path}\n  (expected for '{label}')")
    with open(path) as f:
        return json.load(f)
