"""
nlp_processor.py

spaCy-based NLP utilities:
- Sentence segmentation
- Tokenization
- Lemmatization
- POS tagging

Logic copied 1:1 from clean_data.py.
"""

from typing import Dict, List
import spacy
from spacy.language import Language


NLP: Language | None = None


def get_nlp() -> Language:
    """Lazy-load spaCy model exactly as in the original script."""
    global NLP
    if NLP is None:
        NLP = spacy.load("en_core_web_sm")
    return NLP


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def spacy_process(text: str) -> Dict[str, List[str]]:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    nlp = get_nlp()
    doc = nlp(text)

    return {
        "tokens": [t.text for t in doc],
        "lemmas": [t.lemma_ for t in doc],
        "pos": [t.pos_ for t in doc],
    }
