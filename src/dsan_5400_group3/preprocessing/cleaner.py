"""
cleaner.py

Text cleaning utilities for Wikipedia biography processing.
Contains normalization, removal of references/templates/tables/HTML tags,
Wikipedia boilerplate removal, and whitespace cleanup.

This file preserves the exact logic from clean_data.py (no modifications).
"""

import re
import unicodedata
from pathlib import Path
import pandas as pd


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_control_characters(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", " ", text)


def remove_references_and_tags(text: str) -> str:
    text = re.sub(r"\[\s*[0-9a-zA-Z]+\s*\]", " ", text)
    text = re.sub(r"\[\s*edit\s*\]", " ", text)

    text = re.sub(
        r"<ref[^>]*>.*?</ref>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(r"<[^>]+>", " ", text)

    return text


def remove_templates_and_tables(text: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[\[Category:[^\]]+\]\]", " ", text)
    text = re.sub(
        r"\[\[(File|Image):[^\]]+\]\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return text


def remove_maintenance_phrases(text: str) -> str:
    patterns = [
        r"this article has multiple issues\.[^.]*",
        r"this biography of a living person[^.]*\.",
        r"this article includes a list of general references[^.]*\.",
        r"this biographical article related to [^.]* is a stub \. you can help wikipedia by expanding it \.",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text)
    return text


def clean_text(text):
    """Full cleaning pipeline. Identical to original clean_data.py."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = normalize_unicode(text)
    text = remove_control_characters(text)
    text = remove_references_and_tags(text)
    text = remove_templates_and_tables(text)

    text = text.lower()
    text = remove_maintenance_phrases(text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


class ChunkedCleanerPipeline:
    """
    Chunked text cleaner. This class now ONLY cleans text and produces
    biographies_clean.csv. It no longer performs train/val/test split.
    """

    def __init__(self, raw_csv: Path, processed_dir: Path, chunksize: int = 20000):
        self.raw_csv = Path(raw_csv)
        self.processed_dir = Path(processed_dir)
        self.chunksize = chunksize

        self.cleaned_csv = self.processed_dir / "biographies_clean.csv"

    def run(self):
        print(f"Reading and cleaning raw data in chunks from: {self.raw_csv}")
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Remove old cleaned file if exists
        if self.cleaned_csv.exists():
            self.cleaned_csv.unlink()

        chunks = pd.read_csv(self.raw_csv, chunksize=self.chunksize)
        total_cleaned = 0
        first_chunk = True

        for chunk in chunks:

            # Drop missing pages
            if "missing" in chunk.columns:
                chunk = chunk[chunk["missing"] == False]

            # Ensure text is valid
            chunk["text"] = chunk["text"].fillna("").astype(str)
            chunk = chunk[chunk["text"].str.strip() != ""]

            # Clean
            chunk["text_clean"] = chunk["text"].apply(clean_text)
            chunk = chunk[chunk["text_clean"].str.strip() != ""]

            # Remove original text
            chunk = chunk.drop(columns=["text"])

            # Add simple features
            chunk["article_length_chars"] = chunk["text_clean"].str.len()
            chunk["article_length_words"] = chunk["text_clean"].str.split().str.len()

            # Write chunk to cleaned CSV
            mode = "w" if first_chunk else "a"
            header = first_chunk
            chunk.to_csv(self.cleaned_csv, mode=mode, header=header, index=False)

            total_cleaned += len(chunk)
            first_chunk = False
            print(f"Processed rows so far: {total_cleaned}")

        print(f"\nFinished cleaning. Total cleaned rows: {total_cleaned}")
        print(f"Cleaned dataset saved to: {self.cleaned_csv}")

        return self.cleaned_csv

