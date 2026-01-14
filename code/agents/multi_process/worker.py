import numpy as np

def worker_process(wid, env_fn, out_queue, ctrl_conn, steps_per_batch=200, worker_epsilon=0.05, agent_cfg=None):
    env = env_fn()

    # build local agent from cfg
    from agents.ddqn import DDQNAgent
    agent = DDQNAgent(cfg=agent_cfg, device="cpu")
    agent.q_net.eval()
    agent.target_net.eval()

    # wait initial weights
    while ctrl_conn.poll(0.01):
        msg, payload = ctrl_conn.recv()
        if msg == "weights":
            agent.q_net.load_state_dict(payload["q"])
            agent.target_net.load_state_dict(payload["tgt"])

    while True:
        state = env.reset()
        batch = []

        for _ in range(steps_per_batch):
            # receive latest weights
            if ctrl_conn.poll():
                msg, payload = ctrl_conn.recv()
                if msg == "weights":
                    agent.q_net.load_state_dict(payload["q"])
                    agent.target_net.load_state_dict(payload["tgt"])

            # ✅ FIXED epsilon exploration (your worker_epsilon)
            if np.random.rand() < worker_epsilon:
                action = np.random.randint(agent.n_actions)
            else:
                action = agent.select_action(state, greedy=True)  # pure argmax, no anneal

            next_state, reward, done, info = env.step(action)
            batch.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                state = env.reset()

        out_queue.put(batch)
