from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, List, Dict
import os

import numpy as np
import pandas as pd
import torch


Intent = Literal["buy", "sell"]
Action = Literal["BUY", "SELL", "HOLD"]


@dataclass
class Decision:
    ticker: str
    date: str  # ISO date used for the decision
    action: Action
    confidence: float
    q_values: List[float]
    q_gap: float
    row_index: int


def _softmax_maxprob(q: np.ndarray) -> float:
    q = q.astype(np.float64)
    q = q - np.max(q)
    exp = np.exp(q)
    p = exp / (np.sum(exp) + 1e-12)
    return float(np.max(p))


def _q_gap(q: np.ndarray) -> float:
    if q.size < 2:
        return 0.0
    s = np.sort(q)
    return float(s[-1] - s[-2])


def load_row_meta(meta_parquet: str, meta_csv_fallback: Optional[str] = None) -> pd.DataFrame:
    try:
        df = pd.read_parquet(meta_parquet)
    except Exception as e:
        if meta_csv_fallback and os.path.exists(meta_csv_fallback):
            df = pd.read_csv(meta_csv_fallback)
        else:
            raise RuntimeError(
                f"Failed to read row meta parquet and no valid CSV fallback. "
                f"parquet={meta_parquet} csv={meta_csv_fallback}"
            ) from e

    if "ticker" not in df.columns or "date" not in df.columns:
        raise ValueError(f"row_meta must have columns ['ticker','date'], got {list(df.columns)}")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    return df.reset_index(drop=True)


class ModelAdapter:
    """
    Loads either:
      - a torch.nn.Module checkpoint (callable), OR
      - a dict checkpoint (state_dict / bundle) -> requires model_builder()
    """

    def __init__(self, ckpt_path: str, device: str = "cpu", model_builder=None):
        self.device = torch.device(device)
        self.model = self._load(ckpt_path, model_builder)
        self.model.eval()

    def _extract_state_dict(self, obj):
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict checkpoint, got {type(obj)}")

        # Your bundle format
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]

        # Common bundle formats
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if "policy_net_state_dict" in obj and isinstance(obj["policy_net_state_dict"], dict):
            return obj["policy_net_state_dict"]
        if "q_net_state_dict" in obj and isinstance(obj["q_net_state_dict"], dict):
            return obj["q_net_state_dict"]

        # Raw state_dict heuristic
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj

        raise TypeError(f"Unsupported dict checkpoint keys: {list(obj.keys())[:15]}")


    def _load(self, ckpt_path: str, model_builder):
        obj = torch.load(ckpt_path, map_location=self.device)

        # Case 1: full module saved (best case)
        if callable(obj):
            return obj.to(self.device)

        # Case 2: dict saved (state_dict or bundle)
        if not isinstance(obj, dict):
            raise TypeError(f"Checkpoint type not supported: {type(obj)}")

        if model_builder is None:
            raise RuntimeError(
                "Checkpoint is a dict (state_dict/bundle). "
                "Provide model_builder=lambda: <YourQNetwork(...)> to rebuild the model."
            )

        state_dict = self._extract_state_dict(obj)
        model = model_builder().to(self.device)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys loading {ckpt_path}: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"[WARN] Unexpected keys loading {ckpt_path}: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

        return model

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.model(x)
        q = q.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if q.ndim != 1:
            raise RuntimeError(f"Expected 1D Q-values, got shape={q.shape}")
        return q

class DecisionEngine:
    def __init__(
        self,
        features_path: str,
        row_meta_parquet: str,
        buy_ckpt_path: str,
        sell_ckpt_path: Optional[str] = None,
        row_meta_csv_fallback: Optional[str] = None,
        device: str = "cpu",
        buy_action_map: Optional[Dict[int, Action]] = None,
        sell_action_map: Optional[Dict[int, Action]] = None,
        buy_model_builder=None,          # NEW
        sell_model_builder=None,         # NEW
    ):
        self.X = np.load(features_path).astype(np.float32)
        self.meta = load_row_meta(row_meta_parquet, row_meta_csv_fallback)

        if len(self.X) != len(self.meta):
            raise RuntimeError(f"Alignment mismatch: X rows={len(self.X)} meta rows={len(self.meta)}")

        # CHANGED: pass builder
        self.buy_model = ModelAdapter(buy_ckpt_path, device=device, model_builder=buy_model_builder)
        self.sell_model = (
            ModelAdapter(sell_ckpt_path, device=device, model_builder=sell_model_builder)
            if sell_ckpt_path else None
        )

        self.buy_action_map = buy_action_map or {0: "HOLD", 1: "BUY"}
        self.sell_action_map = sell_action_map or {0: "HOLD", 1: "SELL"}

    def supported_tickers(self) -> List[str]:
        return sorted(self.meta["ticker"].unique().tolist())

    def latest_available_date(self, ticker: str) -> str:
        df = self.meta[self.meta["ticker"] == ticker]
        if df.empty:
            raise ValueError(f"Unsupported ticker: {ticker}")
        return df["date"].iloc[-1].date().isoformat()

    def _select_row(self, ticker: str, as_of: Optional[str]) -> tuple[int, pd.Timestamp]:
        df = self.meta[self.meta["ticker"] == ticker]
        if df.empty:
            raise ValueError(f"Unsupported ticker: {ticker}")

        if as_of is None:
            row = df.iloc[-1]
            return int(row.name), pd.Timestamp(row["date"])

        as_of_ts = pd.to_datetime(as_of)
        df2 = df[df["date"] <= as_of_ts]
        if df2.empty:
            row = df.iloc[0]
            return int(row.name), pd.Timestamp(row["date"])

        row = df2.iloc[-1]
        return int(row.name), pd.Timestamp(row["date"])

    def predict(self, ticker: str, intent: Intent = "buy", as_of: Optional[str] = None) -> Decision:
        row_i, ts = self._select_row(ticker, as_of)
        state = self.X[row_i]

        # --- SELL model expects 15 dims, features artifact provides 12 ---
        if intent == "sell" and state.shape[0] == 12:
            state = np.concatenate([state, np.zeros(3, dtype=np.float32)], axis=0)

        if intent == "buy":
            q = self.buy_model.q_values(state)
            action_idx = int(np.argmax(q))
            action = self.buy_action_map.get(action_idx, "HOLD")
        else:
            if self.sell_model is None:
                raise RuntimeError("sell_ckpt_path not provided but intent='sell' requested")
            q = self.sell_model.q_values(state)
            action_idx = int(np.argmax(q))
            action = self.sell_action_map.get(action_idx, "HOLD")

        conf = _softmax_maxprob(q)
        gap = _q_gap(q)

        return Decision(
            ticker=ticker,
            date=ts.date().isoformat(),
            action=action,
            confidence=conf,
            q_values=[float(v) for v in q.tolist()],
            q_gap=gap,
            row_index=row_i,
        )

    # def recommend_top_k(self, tickers: List[str], k: int = 3, as_of: Optional[str] = None) -> List[Decision]:
    #     decisions = [self.predict(t, intent="buy", as_of=as_of) for t in tickers]

    #     buys = [d for d in decisions if d.action == "BUY"]
    #     buys.sort(key=lambda d: (d.confidence, d.q_gap), reverse=True)

    #     if buys:
    #         return buys[:k]

    #     decisions.sort(key=lambda d: (d.confidence, d.q_gap), reverse=True)
    #     return decisions[:k]
    def recommend_top_k(self, tickers: List[str], k: int = 3, as_of: Optional[str] = None) -> List[Decision]:
        decisions = [self.predict(t, intent="buy", as_of=as_of) for t in tickers]
        decisions.sort(key=lambda d: (d.confidence, d.q_gap), reverse=True)
        return decisions[:k]

