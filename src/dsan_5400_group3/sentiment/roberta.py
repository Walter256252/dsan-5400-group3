"""
roberta.py

Transformer-based sentiment analysis using the CardiffNLP model.
Preserves the exact behavior of run_sentiment.py.
"""

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"


class RobertaSentiment:
    def __init__(self, batch_size=16, max_length=256):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading RoBERTa model '{MODEL_NAME}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        self.label_map = self.model.config.id2label

    def add_roberta_sentiment(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        texts = df[text_col].fillna("").astype(str).tolist()

        labels = []
        confidences = []

        print("[INFO] Running RoBERTa sentiment inference...")

        for i in tqdm(range(0, len(texts), self.batch_size),
                       desc="RoBERTa",
                       ncols=80):

            batch = texts[i : i + self.batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)

            max_probs, pred_ids = torch.max(probs, dim=-1)

            labels.extend(self.label_map[int(pid)] for pid in pred_ids)
            confidences.extend(float(p) for p in max_probs)

        df["roberta_label"] = labels
        df["roberta_confidence"] = confidences

        return df
