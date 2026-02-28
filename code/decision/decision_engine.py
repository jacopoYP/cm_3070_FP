from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, List, Dict

import os

import numpy as np
import pandas as pd
import torch

import torch
import torch.nn as nn

Intent = Literal["should_buy", "should_sell", "recommend_today", "unknown"]
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
    raw_action: str
    raw_action_idx: int

def softmax_maxprob(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    q = q - np.max(q)
    exp = np.exp(q)

    s = exp.sum()
    if not np.isfinite(s) or s == 0.0:
        # degenerate case
        return float(1.0 / q.size)

    p = exp / s
    return float(p.max())

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

        w0 = None
        for name, p in self.model.named_parameters():
            if "net.0.weight" in name:
                w0 = p.detach().cpu().numpy()
                break
        # print("MODEL net.0.weight mean/std:", float(w0.mean()), float(w0.std()))

    def _extract_state_dict(self, obj):
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict checkpoint, got {type(obj)}")

        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]

        # Bundle formats
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if "policy_net_state_dict" in obj and isinstance(obj["policy_net_state_dict"], dict):
            return obj["policy_net_state_dict"]
        if "q_net_state_dict" in obj and isinstance(obj["q_net_state_dict"], dict):
            return obj["q_net_state_dict"]

        # Raw state_dict
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj

        raise TypeError(f"Unsupported dict checkpoint keys: {list(obj.keys())[:15]}")

    def _load(self, ckpt_path: str, model_builder):
        obj = torch.load(ckpt_path, map_location=self.device)

        # Case 1: full module saved
        if callable(obj):
            return obj.to(self.device)

        # Case 2: dict saved
        if not isinstance(obj, dict):
            raise TypeError(f"Checkpoint type not supported: {type(obj)}")

        if model_builder is None:
            raise RuntimeError(
                "Checkpoint is a dict (state_dict/bundle). "
                "Provide model_builder=lambda: <YourQNetwork(...)> to rebuild the model."
            )

        state_dict = self._extract_state_dict(obj)
        model = model_builder().to(self.device)

        model.load_state_dict(state_dict, strict=True)

        k = "net.0.weight"
        if k in state_dict:
            a = state_dict[k].detach().cpu().view(-1)[:5].numpy()
            b = dict(model.named_parameters())[k].detach().cpu().view(-1)[:5].numpy()
            # print("CKPT vs MODEL first5", a, b)

        return model

    def _expected_in_features(self, model: nn.Module) -> int:
        # Works with your MLPQNetwork: model.net is nn.Sequential([Linear, ReLU, ...])
        if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
            for m in model.net:
                if isinstance(m, nn.Linear):
                    return int(m.in_features)
        # Fallback
        for m in model.modules():
            if isinstance(m, nn.Linear):
                return int(m.in_features)
        raise RuntimeError("Could not infer model input dim (no Linear layer found).")

    @torch.no_grad()
    def q_values(self, state):
        expected = self._expected_in_features(self.model)

        s = np.asarray(state, dtype=np.float32).reshape(-1)
        got = int(s.shape[0])

        if got < expected:
            # Pad missing (e.g., SELL needs +3 position features)
            padded = np.zeros((expected,), dtype=np.float32)
            padded[:got] = s
            s = padded
        elif got > expected:
            s = s[:expected]

        x = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)

        # DEBUG: inspect activations of first layer
        with torch.no_grad():
            # DEBUG: walk through Sequential and print activations
            if hasattr(self.model, "net") and isinstance(self.model.net, nn.Sequential):
                h = x
                for i, layer in enumerate(self.model.net):
                    h = layer(h)
                            
                    q = self.model(x)
                    q = q.squeeze(0).detach().cpu().numpy().astype(np.float32)
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

        # Choosing model + action map
        if intent == "buy":
            model = self.buy_model
            # {0:"HOLD", 1:"BUY"}
            action_map = self.buy_action_map
            valid = ("BUY", "HOLD")
        else:
            if self.sell_model is None:
                raise RuntimeError("sell_ckpt_path not provided but intent='sell' requested")
            model = self.sell_model
              # {0:"HOLD", 1:"SELL"}
            action_map = self.sell_action_map
            valid = ("SELL", "HOLD")

        # Q values + raw argmax
        q = model.q_values(state)
        q = np.asarray(q, dtype=np.float32).reshape(-1)

        raw_idx = int(np.argmax(q))
        raw_action = action_map.get(raw_idx, "HOLD")

        if raw_action not in valid:
            raise RuntimeError(f"{intent.upper()} intent produced invalid raw_action={raw_action} idx={raw_idx}")

        # Confidence + margin
        conf = softmax_maxprob(q)

        # If size is minor than 2
        if q.size < 2:
            margin = 0.0
        elif q.size == 2:
            margin = float(abs(float(q[1]) - float(q[0])))
        else:
            top1 = float(np.max(q))
            top2 = float(np.partition(q, -2)[-2])
            margin = float(top1 - top2)

        # Assuming we have HOLD plus one action (BUY or SELL)
        adv = None
        if q.size == 2:
            # infer HOLD idx and ACTION idx from map
            hold_idx = None
            act_idx = None
            for idx, name in action_map.items():
                if name == "HOLD":
                    hold_idx = idx
                elif (intent == "buy" and name == "BUY") or (intent == "sell" and name == "SELL"):
                    act_idx = idx
            if hold_idx is None:
                hold_idx = 0
            if act_idx is None:
                act_idx = 1
            adv = float(q[act_idx] - q[hold_idx])

        # The final action
        final_action = raw_action

        # Margin advises "how separated are actions" whilst advantage tells us "is BUY/Sell actually better than HOLD"

        # Using default margin, might be subjected to GA later
        if intent == "buy" and raw_action == "BUY":
            min_margin = float(getattr(self, "min_buy_gap", 0.0005))
            min_adv = float(getattr(self, "min_buy_adv", 0.0))

            if margin < min_margin:
                final_action = "HOLD"
            elif adv is not None and adv < min_adv:
                final_action = "HOLD"

        if intent == "sell" and raw_action == "SELL":
            min_margin = float(getattr(self, "min_sell_gap", 0.0005))
            min_adv = float(getattr(self, "min_sell_adv", 0.0))

            if margin < min_margin:
                final_action = "HOLD"
            elif adv is not None and adv < min_adv:
                final_action = "HOLD"

        return Decision(
            ticker=ticker,
            date=ts.date().isoformat(),
            action=final_action,
            confidence=float(conf),
            q_values=[float(v) for v in q.tolist()],
            q_gap=float(margin),
            row_index=int(row_i),
            raw_action=str(raw_action),
            raw_action_idx=int(raw_idx),
        )

    # ---------------------------------------------------------------------
    # Best tickers
    # ---------------------------------------------------------------------
    def recommend_top_k(self, tickers: List[str], k: int = 3, as_of: Optional[str] = None) -> List[Decision]:
        decisions = [self.predict(t, intent="buy", as_of=as_of) for t in tickers]

        def buy_adv(d: Decision) -> float:
            # Usual mapping {0:HOLD, 1:BUY}
            if len(d.q_values) >= 2:
                return float(d.q_values[1] - d.q_values[0])
            return 0.0

        decisions.sort(
            key=lambda d: (
                1 if d.raw_action == "BUY" else 0,  # prioritize model BUYs
                buy_adv(d),                          # strongest BUY advantage
                d.confidence,                        # tie-break
            ),
            reverse=True,
        )
        return decisions[:k]

    # ---------------------------------------------------------------------
    # Only for debug purposes
    # ---------------------------------------------------------------------
    def debug_audit_tickers(self, tickers, as_of=None):
        rows = []

        for t in tickers:
            d = self.predict(t, intent="buy", as_of=as_of)

            q_hold = float(d.q_values[0])
            q_buy  = float(d.q_values[1])
            q_gap  = q_buy - q_hold

            # Raw model decision
            raw_argmax = "BUY" if q_buy > q_hold else "HOLD"

            rows.append({
                "ticker": t,
                "final_action": d.action,
                "raw_action": raw_argmax,
                "q_gap": q_gap,
                "q_buy": q_buy,
                "q_hold": q_hold,
                "confidence": float(d.confidence),
            })

        # Sorting 
        rows_sorted = sorted(rows, key=lambda r: r["q_gap"], reverse=True)

        print("\nTICKER  final  raw    q_gap        q_buy      q_hold     conf")
        for r in rows_sorted:
            print(f'{r["ticker"]:>6}  {r["final_action"]:>4}  {r["raw_action"]:>4}  '
                f'{r["q_gap"]:+.6f}  {r["q_buy"]:+.6f}  {r["q_hold"]:+.6f}  '
                f'{r["confidence"]:.3f}')
            
    from typing import Iterable, Optional, List, Dict, Any

    def debug_dump_buy_universe(
        self,
        tickers: Iterable[str],
        as_of: Optional[str] = None,
        sort_by: str = "buy_adv",
        limit: Optional[int] = None,
        show_only_raw_buy: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Print one-line diagnostics per ticker for BUY intent, at a chosen as_of date.

        sort_by: "buy_adv" | "confidence" | "date" | "ticker"
        limit:   print only top N after sorting (None = print all)
        show_only_raw_buy: only keep rows where raw_action == "BUY"
        """
        rows: List[Dict[str, Any]] = []

        for t in tickers:
            d = self.predict(t, intent="buy", as_of=as_of)

            # Determine the actual row/date used
            row_i, ts = self._select_row(t, as_of)

            q_hold = float(d.q_values[0]) if len(d.q_values) > 0 else 0.0
            q_buy  = float(d.q_values[1]) if len(d.q_values) > 1 else 0.0
            buy_adv = q_buy - q_hold

            rows.append({
                "ticker": t,
                "used_date": ts.date().isoformat(),
                "raw": d.raw_action,
                "final": d.action,
                "buy_adv": float(buy_adv),
                "q_hold": q_hold,
                "q_buy": q_buy,
                "conf": float(d.confidence),
                "gap_top2": float(d.q_gap),
                "row_i": int(row_i),
            })

        if show_only_raw_buy:
            rows = [r for r in rows if r["raw"] == "BUY"]

        if sort_by == "buy_adv":
            rows.sort(key=lambda r: (1 if r["raw"] == "BUY" else 0, r["buy_adv"], r["conf"]), reverse=True)
        elif sort_by == "confidence":
            rows.sort(key=lambda r: r["conf"], reverse=True)
        elif sort_by == "date":
            rows.sort(key=lambda r: r["used_date"], reverse=True)
        else:  # ticker
            rows.sort(key=lambda r: r["ticker"])

        if limit is not None:
            rows = rows[:limit]

        print(f"\n=== DUMP (BUY intent) as_of={as_of} ===")
        print("TICKER  used_date    raw   final   buy_adv     q_buy      q_hold     conf   gap2   row_i")
        for r in rows:
            print(
                f'{r["ticker"]:>6}  {r["used_date"]}  '
                f'{r["raw"]:>4}  {r["final"]:>5}  '
                f'{r["buy_adv"]:+.6f}  {r["q_buy"]:+.6f}  {r["q_hold"]:+.6f}  '
                f'{r["conf"]:.3f}  {r["gap_top2"]:.6f}  {r["row_i"]:>5}'
            )

        return rows