import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
import requests

class FinnhubClient:
    def __init__(
        self,
        api_key: str,
        cache_dir: str = "cache/finnhub_news",
        min_interval_s: float = 1.1,   # <= ~54 calls/min
        max_retries: int = 3,
        backoff_s: float = 2.0,
    ):
        import finnhub
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.min_interval_s = float(min_interval_s)
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)

        os.makedirs(self.cache_dir, exist_ok=True)
        self._client = finnhub.Client(api_key=self.api_key)
        self._last_call_ts = 0.0

    def _sleep_if_needed(self) -> None:
        dt = time.time() - self._last_call_ts
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)

    def _cache_path(self, symbol: str, date_from: str, date_to: str) -> str:
        key = f"company_news|{symbol}|{date_from}|{date_to}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{h}.json")

    def _read_cache(self, path: str) -> Optional[Any]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, path: str, data: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def _fetch_company_news_window(self, symbol: str, date_from: str, date_to: str) -> List[Dict[str, Any]]:
        """
        One Finnhub call for a window. Handles throttling + retries.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._sleep_if_needed()
                data = self._client.company_news(symbol, _from=date_from, to=date_to)
                self._last_call_ts = time.time()
                return list(data) if isinstance(data, list) else []
            except Exception as e:
                last_err = e
                print("Error fetching data ", last_err)
                # exponential-ish backoff
                time.sleep(self.backoff_s * (attempt + 1))
        raise RuntimeError(f"company_news failed: {symbol} {date_from}->{date_to}") from last_err

    def company_news_range(
        self,
        symbol: str,
        date_from: str,
        date_to: str,
        window: str = "1Y",      # "6M" also supported
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetches company news using large windows (Yearly by default).
        Saves each window to disk cache so reruns don't hit API again.
        """
        start = pd.to_datetime(date_from).normalize()
        end = pd.to_datetime(date_to).normalize()

        if window not in ("1Y", "6M"):
            raise ValueError("window must be '1Y' or '6M'")

        step = pd.DateOffset(years=1) if window == "1Y" else pd.DateOffset(months=6)

        out: List[Dict[str, Any]] = []
        cur = start

        while cur <= end:
            nxt = (cur + step) - pd.Timedelta(days=1)
            to = min(nxt, end)

            cur_s = cur.date().isoformat()
            to_s = to.date().isoformat()

            cache_path = self._cache_path(symbol, cur_s, to_s)

            if use_cache:
                cached = self._read_cache(cache_path)
                if isinstance(cached, list):
                    print(f"[CACHE] {symbol} {cur_s} -> {to_s}  ({len(cached)})")
                    out.extend(cached)
                    cur = to + pd.Timedelta(days=1)
                    continue

            print(f"[FETCH] {symbol} {cur_s} -> {to_s}")
            data = self._fetch_company_news_window(symbol, cur_s, to_s)
            print("Data: ", data)
            if use_cache:
                self._write_cache(cache_path, data)

            out.extend(data)
            cur = to + pd.Timedelta(days=1)

        # de-duplicate by Finnhub 'id' if present
        seen = set()
        deduped = []
        for item in out:
            _id = item.get("id", None)
            key = _id if _id is not None else (item.get("datetime"), item.get("headline"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped


class AlphaVantageNewsClient:
    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str,
        cache_dir: str = "cache/alphavantage_news",
        min_interval_s: float = 12.5,   # safe for low quotas
        max_retries: int = 3,
        backoff_s: float = 2.0,
        session: Optional[requests.Session] = None,
    ):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.min_interval_s = float(min_interval_s)
        self.max_retries = int(max_retries)
        self.backoff_s = float(backoff_s)
        self._last_call_ts = 0.0

        os.makedirs(self.cache_dir, exist_ok=True)
        self._session = session or requests.Session()

    def _sleep_if_needed(self) -> None:
        dt = time.time() - self._last_call_ts
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)

    @staticmethod
    def _to_av_time(dt: pd.Timestamp) -> str:
        # AlphaVantage expects UTC-like timestamps without timezone: YYYYMMDDTHHMM
        return dt.strftime("%Y%m%dT%H%M")

    def _cache_path(self, tickers: str, time_from: str, time_to: str, limit: int, sort: str) -> str:
        key = f"NEWS_SENTIMENT|tickers={tickers}|time_from={time_from}|time_from={time_to}|limit={limit}|sort={sort}"
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.cache_dir, f"{h}.json")

    def _read_cache(self, path: str) -> Optional[Any]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, path: str, data: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    def fetch_news(
        self,
        tickers: str,
        date_from: str,             # "YYYY-MM-DD" or ISO datetime
        date_to: str,               # "YYYY-MM-DD" or ISO datetime
        limit: int = 1000,
        sort: str = "LATEST",
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        One Alpha Vantage request (NEWS_SENTIMENT) with:
        - tickers
        - time_from (derived from date_from)
        - limit up to 1000
        Cached on disk so reruns don't hit the API.

        Note: We intentionally do NOT set time_to (fetches from date_from to "now").
        """
        # normalize date_from
        dt = pd.to_datetime(date_from)
        if dt.tzinfo is not None:
            dt = dt.tz_convert("UTC").tz_localize(None)
        else:
            # treat naive as UTC; keep consistent & simple
            dt = dt.tz_localize(None)

        end_dt = pd.to_datetime(date_to)
        if dt.tzinfo is not None:
            end_dt = end_dt.tz_convert("UTC").tz_localize(None)
        else:
            # treat naive as UTC; keep consistent & simple
            end_dt = end_dt.tz_localize(None)

        time_from = self._to_av_time(dt)
        time_to = self._to_av_time(end_dt)
        cache_path = self._cache_path(tickers, time_from, time_to, limit, sort)

        if use_cache:
            cached = self._read_cache(cache_path)
            if isinstance(cached, list):
                print(f"[CACHE] {tickers} from {time_from} ({len(cached)})")
                return cached

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": tickers,
            "time_from": time_from,
            "time_to": time_to,
            "limit": str(limit),
            "sort": sort,
            "apikey": self.api_key,
        }

        last_err: Optional[Exception] = None
        # for attempt in range(self.max_retries + 1):
        print("Fetching data for: ", params)
        try:
            self._sleep_if_needed()
            r = self._session.get(self.BASE_URL, params=params, timeout=30)
            self._last_call_ts = time.time()
            r.raise_for_status()

            payload = r.json()
            print(payload)

            # Alpha Vantage error payloads
            if any(k in payload for k in ("Error Message", "Information", "Note")):
                raise RuntimeError(payload)

            data = payload.get("feed", [])
            data = list(data) if isinstance(data, list) else []

            if use_cache:
                self._write_cache(cache_path, data)

            print(f"[FETCH] {tickers} ({len(data)})")
            return data

        except Exception as e:
            last_err = e
            print(f"AlphaVantage error ({tickers}) : {e}")
            # time.sleep(self.backoff_s * (attempt + 1))

        raise RuntimeError(f"NEWS_SENTIMENT failed for {tickers}") from last_err
