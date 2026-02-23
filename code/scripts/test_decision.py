import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from decision.decision_engine import DecisionEngine
from agents.networks import MLPQNetwork
import torch

def required_state_dim(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    w = sd["net.0.weight"]  # first Linear
    return int(w.shape[1])

# DEBUG SELL PT

# import torch
# from pprint import pprint

# ckpt = torch.load("models/sell_agent.pt", map_location="cpu")

# print("Top-level keys:", ckpt.keys())
# print("\n=== cfg type ===", type(ckpt.get("cfg")))
# print("\n=== cfg (raw) ===")
# pprint(ckpt.get("cfg"))

# sd = ckpt["state_dict"]
# print("\nFirst 20 state_dict keys:")
# print(list(sd.keys())[:20])

# print("\nInput dim from net.0.weight:", sd["net.0.weight"].shape[1])


# print("BUY expects:", required_state_dim("models/buy_agent.pt"))
# print("SELL expects:", required_state_dim("models/sell_agent.pt"))

# STATE_DIM_BUY = 12
# STATE_DIM_SELL = 15
# N_ACTIONS = 2

# engine = DecisionEngine(
#     features_path="data/features.npy",
#     row_meta_parquet="data/row_meta.parquet",
#     buy_ckpt_path="models/buy_agent.pt",
#     sell_ckpt_path="models/sell_agent.pt",
#     buy_model_builder=lambda: MLPQNetwork(state_dim=STATE_DIM_BUY, n_actions=N_ACTIONS),
#     sell_model_builder=lambda: MLPQNetwork(state_dim=STATE_DIM_SELL, n_actions=N_ACTIONS),
#     device="cpu",
# )
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/features.npy")
ROW_META_PATH = os.getenv("ROW_META_PATH", "data/row_meta.parquet")
BUY_CKPT_PATH = os.getenv("BUY_CKPT_PATH", "models/buy_agent.pt")
SELL_CKPT_PATH = os.getenv("SELL_CKPT_PATH", "models/sell_agent.pt")
DEVICE = os.getenv("DEVICE", "cpu")

def _ckpt_state_dim(path: str) -> int:
    payload = torch.load(path, map_location="cpu")
    cfg = payload.get("cfg", {}) or {}
    sd = int(cfg.get("state_dim", 0) or 0)
    if sd <= 0:
        raise ValueError(f"Could not infer state_dim from checkpoint cfg: {path}")
    return sd

STATE_DIM_BUY = _ckpt_state_dim(BUY_CKPT_PATH)
STATE_DIM_SELL = _ckpt_state_dim(SELL_CKPT_PATH)
N_ACTIONS = 2

engine = DecisionEngine(
    features_path=FEATURES_PATH,
    row_meta_parquet=ROW_META_PATH,
    buy_ckpt_path=BUY_CKPT_PATH,
    sell_ckpt_path=SELL_CKPT_PATH,
    buy_model_builder=lambda: MLPQNetwork(state_dim=STATE_DIM_BUY, n_actions=N_ACTIONS),
    sell_model_builder=lambda: MLPQNetwork(state_dim=STATE_DIM_SELL, n_actions=N_ACTIONS),
    device=DEVICE,
)

# print("Supported tickers:", engine.supported_tickers())
# print("AAPL latest date:", engine.latest_available_date("AAPL"))

# print(engine.predict("AAPL", intent="buy"))
# print(engine.predict("NVDA", intent="sell"))
# print(engine.predict("MSFT", intent="buy", as_of="2022-06-01"))

# print("Top 3 buys:", engine.recommend_top_k(engine.supported_tickers(), k=3))
# buys = [d for d in decisions if d.action == "BUY"]
# buys_sorted = sorted(buys, key=lambda d: (d.q_gap, d.confidence), reverse=True)
# print("Top 3 BUY decisions:", buys_sorted[:3])
# dates = ["2020-03-20", "2021-11-15", "2022-06-15", "2023-12-29"]


# dates = ["2023-03-15", "2023-07-14", "2023-10-20"]
# tickers = engine.supported_tickers()

# for as_of in dates:
#     print(f"\n=== {as_of} (BUY intent) ===")
#     decisions = [engine.predict(t, intent="buy", as_of=as_of) for t in tickers]

#     # counts
#     from collections import Counter
#     c_final = Counter(d.action for d in decisions)
#     c_raw = Counter(d.raw_action for d in decisions)
#     print("FINAL counts:", dict(c_final))
#     print("RAW counts  :", dict(c_raw))

#     # top raw buys
#     raw_buys = [d for d in decisions if d.raw_action == "BUY"]
#     raw_buys.sort(key=lambda d: (d.q_gap, d.confidence), reverse=True)
#     print("Top 5 RAW BUY:")
#     for d in raw_buys[:5]:
#         print(f"  {d.ticker:5s} final={d.action:4s} raw={d.raw_action:3s} "
#               f"gap={d.q_gap:+.6f} conf={d.confidence:.3f} q={d.q_values}")

# SHOWCASE_DATE = "2023-07-14"
SHOWCASE_DATE = "2022-06-15"

engine.debug_dump_buy_universe(
    engine.supported_tickers(),
    as_of=SHOWCASE_DATE,
    sort_by="buy_adv",
    limit=None,              # print all
    show_only_raw_buy=False  # set True if you only want BUYs
)