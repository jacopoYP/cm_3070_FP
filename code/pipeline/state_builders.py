# pipeline/state_builders.py
from __future__ import annotations
import numpy as np
import pandas as pd

from pipeline.build_dataset import make_state_frame
from features.state_assembler import StateAssembler
from config.system import TradingSystemConfig


def build_states_and_prices(
    ticker: str,
    config: TradingSystemConfig,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Canonical dataset -> rolling state_df -> aligned prices (numpy float32).
    Used by BOTH BuyAgentTrainer and SellAgentTrainer.
    """
    dataset = make_state_frame(ticker, config).dropna()

    feature_cols = [c for c in dataset.columns if c != "price"]
    assembler = StateAssembler(feature_cols=feature_cols, window_size=config.state.window_size)

    state_df = assembler.assemble(dataset)  # pandas DF aligned to rolling window end
    prices = dataset.loc[state_df.index, "price"].values.astype(np.float32)

    assert len(state_df) == len(prices), "State/price alignment mismatch."
    return state_df, prices
