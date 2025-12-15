"""
loader.py

Loads multiple .jsonl files from a directory and merges them into a single
pandas DataFrame or CSV file.
"""

import json
from pathlib import Path
import pandas as pd


class RawJSONLLoader:
    """
    Minimal wrapper around the original JSONL merging logic.
    No logic is changed — only reorganized into a reusable class.
    """

    def __init__(self, input_dir):
        self.input_dir = Path(input_dir)

    def load_all(self):
        """Read all *.jsonl files and return a merged DataFrame."""
        jsonl_files = sorted(self.input_dir.glob("*.jsonl"))

        records = []
        for path in jsonl_files:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))

        return pd.DataFrame.from_records(records)

    def save_csv(self, output_path):
        """Load all JSONL files and save as a CSV, identical to original script."""
        df = self.load_all()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path
