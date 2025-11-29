"""
Summary of this script

Convert multiple .jsonl raw files into a single CSV file.

- All input files are located in:   project_root/data/raw/
- Each input file is a .jsonl file (one JSON object per line).
- The output CSV will be saved to: project_root/data/raw/biographies_raw.csv
"""

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    # 1. Define input and output paths
    project_root = Path(__file__).resolve().parents[1]
    input_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Collect all .jsonl files under raw_data/
    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    # 3. Read each line of each file as a JSON object and collect records
    records = []
    for path in jsonl_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)  # each line should be a JSON object (dict)
                records.append(obj)

    # 4. Convert list of dicts into a DataFrame
    df = pd.DataFrame.from_records(records)

    # 5. Save the merged table as a CSV in data/raw/
    output_csv = output_dir / "biographies_raw.csv"
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()
    print("\ndone.")

