# sentiment/sentiment_filter.py
from __future__ import annotations
from typing import Any
import numpy as np


def check_sentiment(
    x: np.ndarray,
    cfg: Any,
    sentiment_col: int = -2,
    mass_col: int = -1,
) -> bool:
    """
    x: 1D feature vector at time t (shape: [D])
    cfg: config object (expects fields on cfg.trade_manager or cfg directly)
    Assumes sentiment is stored at indices sentiment_col and mass_col.

    Logic:
      - if filter disabled: True
      - if mass < mass_min: True  (no-news => ignore sentiment)
      - else require sentiment >= sentiment_min_score
    """
    # allow cfg to be either cfg.trade_manager or trade_cfg itself
    trade_cfg = getattr(cfg, "trade_manager", cfg)

    if not bool(getattr(trade_cfg, "use_sentiment_filter", False)):
        return True

    mass_min = float(getattr(trade_cfg, "sentiment_mass_min", 0.0))
    sent_min = float(getattr(trade_cfg, "sentiment_min_score", 0.0))

    try:
        sent = float(x[sentiment_col])
        mass = float(x[mass_col])
    except Exception:
        # fail open (don't block buys if feature indexing is wrong)
        return True

    if mass < mass_min:
        return True

    return sent >= sent_min
