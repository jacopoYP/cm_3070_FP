import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def build_sentiment_index(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a daily sentiment index per ticker (EWMA of compound score).
    """
    df_news["compound"] = df_news["headline"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    df_news["date"] = df_news["published"].dt.date
    daily = df_news.groupby(["ticker", "date"])["compound"].mean().reset_index()
    daily["sent_ewma"] = daily.groupby("ticker")["compound"].transform(lambda s: s.ewm(span=3).mean())
    return daily.rename(columns={"date": "datetime"}).set_index("datetime").shift(1)
