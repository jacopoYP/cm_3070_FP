from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from decision.decision_engine import DecisionEngine
from nlp.router import parse_question
from agents.networks import MLPQNetwork

from fastapi.staticfiles import StaticFiles
import os

# Mapping ticker to Company
TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon.com",
    "GOOGL": "Alphabet",
    "GOOGL": "Google",
    "META": "Meta",
    "META": "Facebook",
    "AVGO": "Broadcom",
    "AMD": "Advanced Micro Devices",
    "AMD": "AMD",
    "INTC": "Intel",
    "TSLA": "Tesla.",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble",
    "KO": "Coca-Cola",
    "PEP": "Pepsi",
    "JPM": "JPMorgan Chase",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "BAC": "BOFA",
    "XOM": "Exxon",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "CAT": "Caterpillar",
    "HON": "Honeywell",
    "SPY": "S&P 500",
}

N_ACTIONS = 2

#  Local paths
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/features.npy")
ROW_META_PATH = os.getenv("ROW_META_PATH", "data/row_meta.parquet")
BUY_CKPT_PATH = os.getenv("BUY_CKPT_PATH", "models/buy_agent.pt")
SELL_CKPT_PATH = os.getenv("SELL_CKPT_PATH", "models/sell_agent.pt")
DEVICE = os.getenv("DEVICE", "cpu")

def check_state_dim(path: str) -> int:
    payload = torch.load(path, map_location="cpu")
    cfg = payload.get("cfg", {}) or {}
    sd = int(cfg.get("state_dim", 0) or 0)
    if sd <= 0:
        raise ValueError(f"Could not infer state_dim from checkpoint cfg: {path}")
    return sd

STATE_DIM_BUY = check_state_dim(BUY_CKPT_PATH)
STATE_DIM_SELL = check_state_dim(SELL_CKPT_PATH)

# Decision Engine
engine = DecisionEngine(
    features_path=FEATURES_PATH,
    row_meta_parquet=ROW_META_PATH,
    buy_ckpt_path=BUY_CKPT_PATH,
    sell_ckpt_path=SELL_CKPT_PATH,
    buy_model_builder=lambda: MLPQNetwork(state_dim=STATE_DIM_BUY, n_actions=N_ACTIONS),
    sell_model_builder=lambda: MLPQNetwork(state_dim=STATE_DIM_SELL, n_actions=N_ACTIONS),
    device=DEVICE,
)

# Used to parse the request
class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class DecisionOut(BaseModel):
    ticker: str
    company: Optional[str] = None
    date: str
    action: str
    confidence: float
    q_gap: float
    confidence_label: str

class AskResponse(BaseModel):
    intent: str
    used_date: Optional[str] = None
    decisions: List[DecisionOut]
    note: str

# Standard FastAPI init
app = FastAPI(title="Financial Advisor Bot (Decision API)", version="0.1")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Used to reference for example the css file
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def confidence_label(conf: float) -> str:
    if conf < 0.45:
        return "LOW"
    if conf < 0.55:
        return "MEDIUM"
    return "HIGH"

# ---------------------------------------------------------------------
# Endpoints exposed
# ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    root = Path(__file__).resolve().parents[1]
    html_path = root / "web" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/health")
def health() -> Dict[str, Any]:
    tickers = engine.supported_tickers()
    latest = engine.latest_available_date(tickers[0]) if tickers else None
    return {"ok": True, "tickers": tickers, "latest_date_example": latest}

# List of tickers
@app.get("/tickers")
def tickers() -> Dict[str, Any]:
    supported = engine.supported_tickers()
    return {"supported": supported, "companies": {t: TICKER_TO_COMPANY.get(t) for t in supported}}

# Date used for the demo
DEMO_AS_OF = os.getenv("DEMO_AS_OF", "2023-07-14")

def _help_note(supported: List[str]) -> str:
    examples = [
        "Should I buy AAPL?",
        "Should I sell NVDA?",
        "Which company shares should I buy today?",
        f"Which stocks should I buy on {DEMO_AS_OF}?",
    ]
    return (
        "I can currently answer:\n"
        "1) 'Should I buy <ticker/company>?'\n"
        "2) 'Should I sell <ticker/company>?'\n"
        "3) 'Which company shares should I buy today?'\n\n"
        f"Supported tickers: {supported}\n\n"
        "Examples:\n- " + "\n- ".join(examples) + "\n\n"
        f"Demo note: this deployment uses a fixed evaluation date (as_of={DEMO_AS_OF})."
    )

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    parsed = parse_question(req.question, ticker_to_company=TICKER_TO_COMPANY, default_top_k=req.top_k or 3)
    supported_list = engine.supported_tickers()
    supported = set(supported_list)

    # Handle recommend intent explicitly
    if parsed.intent == "recommend_today":
        k = int(req.top_k or getattr(parsed, "top_k", None) or 3)
        picks = engine.recommend_top_k(supported_list, k=k, as_of=DEMO_AS_OF)

        out = [
            DecisionOut(
                ticker=d.ticker,
                company=TICKER_TO_COMPANY.get(d.ticker),
                date=d.date,
                action=d.action,
                confidence=float(d.confidence),
                q_gap=float(d.q_gap),
                confidence_label=confidence_label(float(d.confidence)),
            )
            for d in picks
        ]
        used_date = out[0].date if out else DEMO_AS_OF
        return AskResponse(
            intent="recommend_today",
            used_date=used_date,
            decisions=out,
            note=(
                f"Decisions are based on dataset date {used_date} (demo as_of={DEMO_AS_OF}) "
            ),
        )

    # Only allow buy/sell intents
    if parsed.intent not in ("should_buy", "should_sell"):
        return AskResponse(
            intent=parsed.intent,
            used_date=None,
            decisions=[],
            note=_help_note(supported_list),
        )

    # Validate ticker for buy/sell questions
    ticker = (parsed.ticker or "").strip().upper()
    if not ticker:
        return AskResponse(
            intent=parsed.intent,
            used_date=None,
            decisions=[],
            note=_help_note(supported_list),
        )

    if ticker not in supported:
        return AskResponse(
            intent=parsed.intent,
            used_date=None,
            decisions=[],
            note=(
                f"Ticker '{ticker}' is not currently supported.\n\n"
                + _help_note(supported_list)
            ),
        )

    # Prediction with explicit intent mapping
    if parsed.intent == "should_buy":
        d = engine.predict(ticker, intent="buy", as_of="2023-07-14")
    else:
        d = engine.predict(ticker, intent="sell", as_of="2023-07-14")
        if d.action == "SELL":
            # Forcing HOLD
            d = d.__class__(**{**d.__dict__, "action": "HOLD"})  
    
    return AskResponse(
        intent=parsed.intent,
        used_date=d.date,
        decisions=[
            DecisionOut(
                ticker=d.ticker,
                company=TICKER_TO_COMPANY.get(d.ticker),
                date=d.date,
                action=d.action,
                confidence=float(d.confidence),
                q_gap=float(d.q_gap),
                confidence_label=confidence_label(float(d.confidence))
            )
        ],
        note=(
            f"Decision is based on dataset date {d.date} (demo as_of={DEMO_AS_OF})"
        ),
    )