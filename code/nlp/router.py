from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Literal, Dict

# ---------------------------------------------------------------------
# Supported intents
# ---------------------------------------------------------------------
Intent = Literal[
    "should_buy",
    "should_sell",
    "recommend_today",
    "unknown",
]

@dataclass
class ParsedQuery:
    intent: Intent
    ticker: Optional[str] = None
    top_k: int = 3
    normalized_text: str = ""


# ---------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------
def _clean_text(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _build_alias_map(ticker_to_company: Dict[str, str]) -> Dict[str, str]:
    """
    Build alias → ticker map.
    Example:
        "apple" → "AAPL"
        "aapl"  → "AAPL"
    """
    alias = {}

    for ticker, company in ticker_to_company.items():
        tkr_u = ticker.upper()

        # ticker aliases
        alias[ticker.lower()] = tkr_u
        alias[tkr_u.lower()] = tkr_u

        # company name alias
        if company:
            name = company.strip().lower()
            alias[name] = tkr_u

            # Remove punctuation
            name2 = re.sub(r"[^\w\s]", "", name)

            # Remove most common suffixes
            name2 = re.sub(
                r"\b(inc|incorporated|corp|corporation|ltd|limited|plc|co|company|class)\b",
                "",
                name2,
            )

            name2 = re.sub(r"\s+", " ", name2).strip()

            if name2 and name2 != name:
                alias[name2] = tkr_u

    return alias


def extract_ticker(text_norm: str, alias_map: Dict[str, str]) -> Optional[str]:
    # Extract ticker from normalized text.

    m = re.search(r"\$([a-z]{1,6})\b", text_norm)
    if m:
        candidate = m.group(1).upper()
        if candidate in set(alias_map.values()):
            return candidate

    # Token scan
    tokens = re.findall(r"[a-z0-9]+", text_norm)
    for tok in tokens:
        if tok in alias_map:
            return alias_map[tok]

    # Substring scan
    for alias, ticker in alias_map.items():
        if len(alias) >= 4 and alias in text_norm:
            return ticker

    return None


# ---------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------
def parse_question(
    question: str,
    ticker_to_company: Dict[str, str],
    default_top_k: int = 3,
) -> ParsedQuery:
    """
    Examples:

      - "Should I buy Apple?"
      - "Should I sell NVDA?"
      - "Which company shares should I buy today?"
      - "What should I buy?"
    """
    
    # Cleaning the text 
    t = _clean_text(question)
    alias_map = _build_alias_map(ticker_to_company)

    # RECOMMEND TODAY
    if re.search(r"\b(which|what)\b.*\b(stocks|shares|company|companies)\b.*\b(buy|purchase)\b", t):
        return ParsedQuery(
            intent="recommend_today",
            top_k=default_top_k,
            normalized_text=t,
        )

    if re.search(r"\bwhat should i buy\b", t):
        return ParsedQuery(
            intent="recommend_today",
            top_k=default_top_k,
            normalized_text=t,
        )

    if re.search(r"\b(which|what)\b.*\b(buy|purchase)\b", t) and (
        "today" in t or "now" in t
    ):
        return ParsedQuery(
            intent="recommend_today",
            top_k=default_top_k,
            normalized_text=t,
        )

    # Should Buy?
    if "buy" in t:
        ticker = extract_ticker(t, alias_map)

        if re.search(r"\bshould i\b.*\bbuy\b", t) or ticker:
            return ParsedQuery(
                intent="should_buy",
                ticker=ticker,
                normalized_text=t,
            )

    # Should Sell?
    if "sell" in t:
        ticker = extract_ticker(t, alias_map)

        if re.search(r"\bshould i\b.*\bsell\b", t) or ticker:
            return ParsedQuery(
                intent="should_sell",
                ticker=ticker,
                normalized_text=t,
            )

    # Fallback
    return ParsedQuery(
        intent="unknown",
        top_k=default_top_k,
        normalized_text=t,
    )