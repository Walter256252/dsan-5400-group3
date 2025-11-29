"""
Summary of this script

- Load data/raw/biographies_raw.csv in chunks
- Clean the text column:
    * lowercasing
    * removing markup, references, tables, templates, HTML-like tags
    * handling weird unicode/control characters
- Remove common Wikipedia maintenance templates
- Add simple sanity-check features (length)
- Save the cleaned dataset to data/processed/biographies_clean.csv
- Split the cleaned data into train/val/test (80/10/10) and save to:
    * data/processed/train.csv
    * data/processed/val.csv
    * data/processed/test.csv
- sentence splitting
- spaCy-based tokenization, lemmatization, and POS tagging
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import spacy
from spacy.language import Language


# 1. Text cleaning utilities
# Normalize unicode characters to NFKC form
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# Remove non-printable ASCII control characters
def remove_control_characters(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", " ", text)


# Remove common Wikipedia reference markers and HTML-like tags.
def remove_references_and_tags(text: str) -> str:
    """
    Remove common Wikipedia reference markers and HTML-like tags.

    This includes:
    - [1], [23], [a], [ 1 ], [ edit ], etc.
    - <ref> ... </ref> blocks
    - Generic <tag> ... </tag> or self-closing <tag />
    """
    # [1], [23], [a], [ 1 ], [ edit ] etc. (allow optional spaces)
    text = re.sub(r"\[\s*[0-9a-zA-Z]+\s*\]", " ", text)
    text = re.sub(r"\[\s*edit\s*\]", " ", text)

    # <ref> ... </ref> (dot matches newline)
    text = re.sub(
        r"<ref[^>]*>.*?</ref>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Any remaining HTML-like tags <...>
    text = re.sub(r"<[^>]+>", " ", text)

    return text


# Remove simple Wikipedia templates and tables.
def remove_templates_and_tables(text: str) -> str:
    # Remove {{ ... }} templates (non-nested)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)

    # Remove tables starting with '{|' and ending with '|}'
    text = re.sub(
        r"\{\|.*?\|\}",
        " ",
        text,
        flags=re.DOTALL,
    )

    # Remove category lines like [[Category:Something]]
    text = re.sub(r"\[\[Category:[^\]]+\]\]", " ", text)

    # Remove File/Image links like [[File:...]] or [[Image:...]]
    text = re.sub(
        r"\[\[(File|Image):[^\]]+\]\]",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    return text


# Remove common Wikipedia maintenance templates that are not about the person
def remove_maintenance_phrases(text: str) -> str:
    """
    Remove standard maintenance sentences such as:
    - "this article has multiple issues..."
    - "this biography of a living person..."
    - "this article includes a list of general references..."
    - "this biographical article related to ... is a stub. you can help wikipedia..."
    Assumes the input text is already lowercased.
    """
    patterns = [
        r"this article has multiple issues\.[^.]*",
        r"this biography of a living person[^.]*\.",
        r"this article includes a list of general references[^.]*\.",
        r"this biographical article related to [^.]* is a stub \. you can help wikipedia by expanding it \.",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text)
    return text


# Full cleaning pipeline for raw Wikipedia biography text
def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Normalize unicode
    text = normalize_unicode(text)

    # Remove control characters
    text = remove_control_characters(text)

    # Remove references and tags
    text = remove_references_and_tags(text)

    # Remove templates and tables
    text = remove_templates_and_tables(text)

    # Lowercase (so that regex patterns for maintenance phrases match)
    text = text.lower()

    # Remove maintenance templates / boilerplate sentences
    text = remove_maintenance_phrases(text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 2. Sentence splitting utility

# Lazy-load spaCy model to avoid loading it when not needed
NLP: Language | None = None


def get_nlp() -> Language:
    """Lazy-load the spaCy English model."""
    global NLP
    if NLP is None:
        NLP = spacy.load("en_core_web_sm")
    return NLP


# Split cleaned text into sentences using spaCy's sentence segmentation
def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


# 3. spaCy-based tokenization, lemmatization, POS tagging
def spacy_process(text: str) -> Dict[str, List[str]]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    nlp = get_nlp()
    doc = nlp(text)

    tokens = [t.text for t in doc]
    lemmas = [t.lemma_ for t in doc]
    pos_tags = [t.pos_ for t in doc]

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "pos": pos_tags,
    }


# 4. Main cleaning + split pipeline  (chunked version)
def main() -> None:
    # Define paths
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    raw_csv = raw_dir / "biographies_raw.csv"
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv = processed_dir / "biographies_clean.csv"

    print(f"Reading and cleaning raw data in chunks from: {raw_csv}")

    # If an old cleaned file exists, remove it so we can append safely
    if cleaned_csv.exists():
        cleaned_csv.unlink()

    # Process the raw CSV in chunks to save memory
    chunks = pd.read_csv(raw_csv, chunksize=20000)
    total_cleaned = 0
    first_chunk = True

    for chunk in chunks:
        # Filter out missing pages if the column exists
        if "missing" in chunk.columns:
            chunk = chunk[chunk["missing"] == False]

        # Ensure text column is string and non-empty
        chunk["text"] = chunk["text"].fillna("").astype(str)
        chunk = chunk[chunk["text"].str.strip() != ""]

        # Clean text
        chunk["text_clean"] = chunk["text"].apply(clean_text)

        # Drop rows where cleaned text is empty
        chunk = chunk[chunk["text_clean"].str.strip() != ""]

        # Drop original text column to save memory
        chunk = chunk.drop(columns=["text"])

        # Basic sanity features: length
        chunk["article_length_chars"] = chunk["text_clean"].str.len()
        chunk["article_length_words"] = chunk["text_clean"].str.split().str.len()

        # Append this cleaned chunk to the output CSV
        mode = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(cleaned_csv, mode=mode, header=header, index=False)

        total_cleaned += len(chunk)
        first_chunk = False
        print(f"Processed rows so far: {total_cleaned}")

    print(f"Finished chunked cleaning. Total cleaned rows: {total_cleaned}")
    print(f"Cleaned dataset saved to: {cleaned_csv}")

    # Now load the cleaned dataset once for splitting (one text column only)
    df = pd.read_csv(cleaned_csv)
    print("Loaded cleaned data for splitting, shape:", df.shape)

    # Train/Val/Test split (80/10/10)
    print("Creating train/val/test splits (80/10/10)...")
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n = len(df_shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train + n_val].copy()
    df_test = df_shuffled.iloc[n_train + n_val:].copy()

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print("Train size:", len(df_train))
    print("Val size:", len(df_val))
    print("Test size:", len(df_test))
    print("Done.")


if __name__ == "__main__":
    main()
