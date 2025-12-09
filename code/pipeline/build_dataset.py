import pandas as pd
from pipeline.fetch_data import fetch_ohlcv
from features.technicals import compute_indicators
from features.scaling import rolling_zscore, winsorize
from sentiment.scoring import build_sentiment_index

def make_state_frame(ticker: str, cfg: dict, news_df=None):
    df = fetch_ohlcv(ticker, cfg["start_date"], cfg["end_date"])
    # print(df)
    tech = compute_indicators(df)
    features = tech.copy()

    # # Add sentiment if available
    if news_df is not None:
        sent = build_sentiment_index(news_df[news_df["ticker"] == ticker])
        features = features.join(sent, how="left").fillna(0)

    # # Normalize safely
    features = rolling_zscore(features, window=cfg["scaling"]["window"])
    features = winsorize(features)
    features = features.dropna()

    # # Combine with adjusted close for RL environment
    state = features.copy()
    state["price"] = df["adj close"].reindex(state.index)
    # state["price"] = df["close"].reindex(state.index)
    return state
