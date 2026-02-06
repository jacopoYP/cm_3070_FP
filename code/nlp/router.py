from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Literal, Dict, List

Intent = Literal["should_buy", "should_sell", "recommend_today"]

@dataclass
class ParsedQuery:
    intent: Intent
    ticker: Optional[str] = None      # for should_buy/should_sell
    top_k: int = 3                    # for recommend_today
    normalized_text: str = ""


def _clean_text(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _build_alias_map(ticker_to_company: Dict[str, str]) -> Dict[str, str]:
    """
    Builds alias -> ticker mapping (e.g. "apple" -> "AAPL", "aapl" -> "AAPL")
    """
    alias = {}
    for tkr, name in ticker_to_company.items():
        tkr_u = tkr.upper()
        alias[tkr.lower()] = tkr_u
        alias[tkr_u.lower()] = tkr_u

        if name:
            n = name.strip().lower()
            alias[n] = tkr_u

            # add simple variants (remove punctuation, "inc", "corp", "ltd", etc.)
            n2 = re.sub(r"[^\w\s]", "", n)
            n2 = re.sub(r"\b(inc|incorporated|corp|corporation|ltd|limited|plc|co|company|class)\b", "", n2)
            n2 = re.sub(r"\s+", " ", n2).strip()
            if n2 and n2 != n:
                alias[n2] = tkr_u

    # Add a couple of common nicknames manually if in universe
    # (optional; keep minimal)
    if "GOOGL" in ticker_to_company:
        alias["google"] = "GOOGL"
        alias["alphabet"] = "GOOGL"

    return alias


def extract_ticker(text_norm: str, alias_map: Dict[str, str]) -> Optional[str]:
    """
    Try to find a supported ticker/company mention in the normalized text.
    Strategy:
      1) check for $TICKER like $AAPL
      2) token-based scan for exact alias matches
      3) substring scan for company names (safe because universe is small)
    """
    m = re.search(r"\$([a-z]{1,6})\b", text_norm)
    if m:
        cand = m.group(1).upper()
        # validate cand exists in alias_map values
        if cand in set(alias_map.values()):
            return cand

    tokens = re.findall(r"[a-z0-9]+", text_norm)
    for tok in tokens:
        if tok in alias_map:
            return alias_map[tok]

    # substring pass (for multi-word names like "microsoft corporation")
    for a, tkr in alias_map.items():
        if len(a) >= 4 and a in text_norm:
            return tkr

    return None


def parse_question(
    question: str,
    ticker_to_company: Dict[str, str],
    default_top_k: int = 3,
) -> ParsedQuery:
    """
    Supports 3 core intents:
      - Which company shares should I buy today?  -> recommend_today
      - Should I buy Apple stocks?                -> should_buy
      - Should I sell Nvidia?                     -> should_sell
    """
    t = _clean_text(question)
    alias_map = _build_alias_map(ticker_to_company)

    # intent detection (simple rules)
    # recommend_today
    if re.search(r"\b(which|what)\b.*\b(buy|purchase)\b", t) and ("today" in t or "now" in t):
        return ParsedQuery(intent="recommend_today", top_k=default_top_k, normalized_text=t)

    if re.search(r"\bwhat should i buy\b", t) or re.search(r"\bwhich (shares|stocks)\b.*\bbuy\b", t):
        # treat as recommend; "today" implied
        return ParsedQuery(intent="recommend_today", top_k=default_top_k, normalized_text=t)

    # should_buy / should_sell
    if re.search(r"\bshould i\b.*\bbuy\b", t) or re.search(r"\bshould i buy\b", t) or re.search(r"\bbuy\b.*\b(aapl|msft|nvda|amzn|googl)\b", t):
        ticker = extract_ticker(t, alias_map)
        return ParsedQuery(intent="should_buy", ticker=ticker, normalized_text=t)

    if re.search(r"\bshould i\b.*\bsell\b", t) or re.search(r"\bshould i sell\b", t) or re.search(r"\bsell\b.*\b(aapl|msft|nvda|amzn|googl)\b", t):
        ticker = extract_ticker(t, alias_map)
        return ParsedQuery(intent="should_sell", ticker=ticker, normalized_text=t)

    # fallback: if it contains "sell" -> should_sell, else if contains "buy" -> should_buy
    if "sell" in t:
        return ParsedQuery(intent="should_sell", ticker=extract_ticker(t, alias_map), normalized_text=t)
    if "buy" in t:
        return ParsedQuery(intent="should_buy", ticker=extract_ticker(t, alias_map), normalized_text=t)

    # default: recommend_today
    return ParsedQuery(intent="recommend_today", top_k=default_top_k, normalized_text=t)
