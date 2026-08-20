"""Small JSONL helpers for the standalone RULER dataset generators."""

import json
from pathlib import Path


def write_manifest(path, records, ensure_ascii=True):
    """Write one JSON object per line without requiring NeMo."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=ensure_ascii,
                )
            )
            output.write("\n")
