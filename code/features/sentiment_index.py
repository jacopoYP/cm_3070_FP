from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from data.finnhub_api import FinnhubClient
# from sentiment.finbert import add_finbert_sentimen
from core.math_utils import MIN_TEMP


# Basic finance-ish lexicon.
POS_WORDS = {
    "beat", "beats", "growth", "surge", "record", "profit", "profits", "upgrade",
    "strong", "bull", "bullish", "outperform", "rally", "upside", "win", "wins"
}
NEG_WORDS = {
    "miss", "misses", "drop", "falls", "plunge", "loss", "losses", "downgrade",
    "weak", "bear", "bearish", "underperform", "crash", "downside", "lawsuit"
}

def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    
    # simple tokenizer, to be imprved later in case
    t = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [w for w in t.split() if len(w) >= 2]


def headline_polarity(headline: str) -> float:
    toks = _tokenize(headline)
    if not toks:
        return 0.0
    pos = sum(1 for w in toks if w in POS_WORDS)
    neg = sum(1 for w in toks if w in NEG_WORDS)

    # Normalizing polarity in [-1, 1]
    den = pos + neg
    return float((pos - neg) / den) if den > 0 else 0.0


@dataclass
class SentimentIndexConfig:
    ewma_alpha: float = 0.2
    clip: float = 3.0
    z_window: int = 60
    use_zscore: bool = True


class SentimentIndex:
    """
    Build a daily sentiment series aligned to the price index.
    - Pull Finnhub company news per day
    - Aggregate to a single score per day
    - EWMA smooth
    - Optional rolling z-score
    """

    def __init__(self, client: FinnhubClient, cfg: Optional[SentimentIndexConfig] = None):
        self.client = client
        self.cfg = cfg or SentimentIndexConfig()

    def build_daily_scores(self, symbol: str, dates: List[str]) -> np.ndarray:
        scores = np.zeros((len(dates),), dtype=np.float32)

        for i, d in enumerate(dates):
            day_from = d
            day_to = d
            news = self.client.company_news(symbol, day_from, day_to)

            if not news:
                scores[i] = 0.0
                continue

            pols = []
            for item in news:
                h = item.get("headline") or ""
                pols.append(headline_polarity(h))

            scores[i] = float(np.mean(pols)) if pols else 0.0

        return scores

    def ewma(self, x: np.ndarray) -> np.ndarray:
        a = float(self.cfg.ewma_alpha)
        out = np.zeros_like(x, dtype=np.float32)
        m = 0.0
        for i in range(len(x)):
            m = a * float(x[i]) + (1.0 - a) * m
            out[i] = m
        return out

    def rolling_z(self, x: np.ndarray) -> np.ndarray:
        w = int(self.cfg.z_window)
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            win = x[lo : i + 1]
            mu = float(np.mean(win))
            sd = float(np.std(win)) + MIN_TEMP
            z = (float(x[i]) - mu) / sd
            if self.cfg.clip is not None:
                z = float(np.clip(z, -self.cfg.clip, self.cfg.clip))
            out[i] = z
        return out

    def build_feature(self, symbol: str, dates: List[str]) -> np.ndarray:
        raw = self.build_daily_scores(symbol, dates)
        sm = self.ewma(raw)
        if self.cfg.use_zscore:
            return self.rolling_z(sm)
        return sm.astype(np.float32)
