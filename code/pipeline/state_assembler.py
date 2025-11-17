import pandas as pd

class StateAssembler:
    """
    A simple helper that turns engineered features into the final
    state vector used by the RL environments.
    """

    def __init__(self, feature_cols, window_size=30):
        self.feature_cols = feature_cols
        self.window_size = window_size

    def assemble(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build rolling window states. Each state is a 2D array
        (window_size × N_features) later flattened for the agent.
        """
        # Keep only clean feature columns
        data = df[self.feature_cols].copy()

        # Rolling window construction
        states = []
        for i in range(self.window_size, len(data)):
            window = data.iloc[i - self.window_size : i].values
            states.append(window.flatten())

        # Align with original index (drop first window_size rows)
        state_df = pd.DataFrame(
            states,
            index=data.index[self.window_size:],
        )

        return state_df
