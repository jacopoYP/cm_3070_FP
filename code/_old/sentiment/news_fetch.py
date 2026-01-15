import requests
import pandas as pd

def fetch_news(ticker: str, api_key: str, days: int = 7) -> pd.DataFrame:
    """
    Fetch recent news for a ticker using NewsAPI or similar.
    """
    url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&apiKey={api_key}"
    resp = requests.get(url)
    data = resp.json().get("articles", [])
    df = pd.DataFrame([{
        "ticker": ticker,
        "published": a["publishedAt"],
        "headline": a["title"]
    } for a in data])
    df["published"] = pd.to_datetime(df["published"])
    return df
