import multiprocessing as mp
import numpy as np

def worker_process(
    wid: int,
    env_fn,
    agent_fn,
    out_queue: mp.Queue,
    steps_per_batch: int = 200,
):
    """
    A worker that:
    - Instantiates its own environment + agent
    - Runs for 'steps_per_batch'
    - Pushes transitions to the central queue
    """

    env = env_fn()
    agent = agent_fn()

    while True:
        state = env.reset()
        batch = []

        for _ in range(steps_per_batch):
            # Greedy? No — training: use epsilon
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Store raw transition
            batch.append((state, action, reward, next_state, done))

            state = next_state

            if done:
                state = env.reset()

        # Send all transitions to main process
        out_queue.put(batch)
