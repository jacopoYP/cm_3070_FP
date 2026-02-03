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

STATE_DIM_BUY = 12
STATE_DIM_SELL = 15
N_ACTIONS = 2

engine = DecisionEngine(
    features_path="data/features.npy",
    row_meta_parquet="data/row_meta.parquet",
    buy_ckpt_path="models/buy_agent.pt",
    sell_ckpt_path="models/sell_agent.pt",
    buy_model_builder=lambda: MLPQNetwork(state_dim=STATE_DIM_BUY, n_actions=N_ACTIONS),
    sell_model_builder=lambda: MLPQNetwork(state_dim=STATE_DIM_SELL, n_actions=N_ACTIONS),
    device="cpu",
)

print("Supported tickers:", engine.supported_tickers())
print("AAPL latest date:", engine.latest_available_date("AAPL"))

print(engine.predict("AAPL", intent="buy"))
print(engine.predict("NVDA", intent="sell"))
print(engine.predict("MSFT", intent="buy", as_of="2022-06-01"))

print("Top 3 buys:", engine.recommend_top_k(engine.supported_tickers(), k=3))
