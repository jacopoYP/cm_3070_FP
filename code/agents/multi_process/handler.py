from agents.ddqn import DDQNAgent  # adjust import path if needed

class EnvHandler:
    def __init__(
        self,
        state_df,
        prices,
        EnvClass,
        horizon,
        transaction_cost,
        min_steps_before_sell=None,
        lambda_dd=None,
        lambda_vol=None,
        hold_penalty_long=None,
    ):
        self.state_df = state_df
        self.prices = prices
        self.EnvClass = EnvClass
        self.horizon = horizon
        self.transaction_cost = transaction_cost

        self.min_steps_before_sell = min_steps_before_sell
        self.lambda_dd = lambda_dd
        self.lambda_vol = lambda_vol
        self.hold_penalty_long = hold_penalty_long

    def __call__(self):
        # If SellEnv
        if "SellEnv" in self.EnvClass.__name__:
            return self.EnvClass(
                state_window_df=self.state_df,
                price_series=self.prices,
                horizon=self.horizon,
                transaction_cost=self.transaction_cost,
                min_steps_before_sell=self.min_steps_before_sell,
                lambda_dd=self.lambda_dd,
                lambda_vol=self.lambda_vol,
                hold_penalty_long=self.hold_penalty_long,
            )

        # If BuyEnv
        return self.EnvClass(
            state_window_df=self.state_df,
            price_series=self.prices,
            horizon=self.horizon,
            transaction_cost=self.transaction_cost,
        )


class AgentHandler:
    def __init__(self, agent_template: DDQNAgent):
        self.template = agent_template

    def __call__(self):
        A = self.template

        # Rebuild a fresh agent with the same hyperparameters
        return DDQNAgent(
            state_dim=A.state_dim,
            n_actions=A.n_actions,
            gamma=A.gamma,
            lr=A.lr,
            batch_size=A.batch_size,
            buffer_size=A.buffer_size,
            target_update_freq=A.target_update_freq,
            epsilon_start=A.epsilon_start,
            epsilon_end=A.epsilon_end,
            epsilon_decay_steps=A.epsilon_decay_steps,
            device=A.device.type,  # 'cpu' or 'cuda'
        )

