from __future__ import annotations

import numpy as np
from typing import Iterable, List
from transformers import pipeline

# Load once (global, cached by HF)
_sent_pipe = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    truncation=True,
)


def finbert_score(headlines: Iterable[str], batch_size: int = 16) -> np.ndarray:
    """
    Returns sentiment score per headline:
        score = P(positive) - P(negative)

    Range: [-1, +1]
    """
    headlines = list(headlines)
    if len(headlines) == 0:
        return np.empty((0,), dtype=np.float32)

    outputs = _sent_pipe(headlines, batch_size=batch_size)

    scores: List[float] = []
    for out in outputs:
        d = {x["label"].lower(): float(x["score"]) for x in out}
        pos = d.get("positive", 0.0)
        neg = d.get("negative", 0.0)
        scores.append(pos - neg)

    return np.asarray(scores, dtype=np.float32)


def add_finbert_sentiment(news_items: list[dict]) -> list[dict]:
    """
    Mutates Finnhub news items by adding:
        item["sentiment"] = FinBERT score
    """
    if not news_items:
        return news_items

    headlines = [item.get("headline", "") for item in news_items]
    scores = finbert_score(headlines)

    for item, score in zip(news_items, scores):
        item["sentiment"] = float(score)

    return news_items
