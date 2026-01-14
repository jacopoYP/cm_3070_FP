# agents/multi_process/multi_process_trainer.py
import multiprocessing as mp
from agents.multi_process.worker import worker_process
import numpy as np
import torch

class MultiProcessTrainer:
    def __init__(
        self,
        agent,                 # central learner (DDQNAgent)
        env_fn,                # EnvHandler (picklable)
        n_workers: int = 4,
        steps_per_batch: int = 200,
        sync_every: int = 10,
        log_every: int = 10,
        worker_epsilon: float = 0.05,
    ):
        self.agent = agent
        self.env_fn = env_fn
        self.n_workers = n_workers
        self.steps_per_batch = steps_per_batch
        self.sync_every = sync_every
        self.log_every = log_every
        self.worker_epsilon = worker_epsilon

        import multiprocessing as mp
        self.queue = mp.Queue(maxsize=n_workers * 2)
        self.workers = []
        self.ctrl_pipes = []

    def start_workers(self):
        import multiprocessing as mp

        self.ctrl_pipes = []
        self.workers = []

        for i in range(self.n_workers):
            parent_conn, child_conn = mp.Pipe()
            self.ctrl_pipes.append(parent_conn)

            p = mp.Process(
                target=worker_process,
                args=(
                    i,
                    self.env_fn,
                    self.queue,
                    child_conn,
                    self.steps_per_batch,
                    self.worker_epsilon,
                    self.agent.cfg,   # ✅ pass cfg so worker can build same QNetwork
                ),
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def broadcast_weights(self):
        payload = {
            "q": {k: v.detach().cpu() for k, v in self.agent.q_net.state_dict().items()},
            "tgt": {k: v.detach().cpu() for k, v in self.agent.target_net.state_dict().items()},
        }
        for conn in self.ctrl_pipes:
            try:
                conn.send(("weights", payload))
            except (BrokenPipeError, EOFError):
                pass

    #     print("[MP Trainer] Training complete.")
    def train(self, n_batches: int = 2000, updates_per_batch: int = 50, log_every: int | None = None, probe_every: int = 50):
        print("[MP Trainer] Starting workers...")
        self.start_workers()
        self.broadcast_weights()

        if log_every is None:
            log_every = self.log_every

        for batch_idx in range(n_batches):
            batch = self.queue.get()

            for (s, a, r, ns, d) in batch:
                self.agent.push_transition(s, a, r, ns, d)

            for _ in range(updates_per_batch):
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    self.agent.update()

            if (batch_idx + 1) % self.sync_every == 0:
                self.broadcast_weights()

            if (batch_idx + 1) % log_every == 0:
                print(f"[Batch {batch_idx+1}/{n_batches}] Buffer size={len(self.agent.replay_buffer)} | eps={self.agent.epsilon:.3f}")

            # ✅ Probe: quick confidence stats on a few fixed states
            if (batch_idx + 1) % probe_every == 0:
                stats = self._probe_confidence(n_states=64, temp=1.0)
                print(
                    f"[Probe @ batch {batch_idx+1}] "
                    f"BUY_conf mean={stats['mean']:.4f} max={stats['max']:.4f} min={stats['min']:.4f} "
                    f"| greedy_buy_rate={stats['greedy_buy_rate']:.3f}"
                )

        print("[MP Trainer] Training complete.")

    def _probe_confidence(self, n_states: int = 64, temp: float = 1.0) -> dict:
        """
        Sample a few states from the replay buffer and compute:
        - BUY probability via softmax(Q/temp)
        - greedy buy rate (argmax == BUY)
        """
        if len(self.agent.replay_buffer) < max(100, n_states):
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "greedy_buy_rate": 0.0}

        # sample transitions from buffer (no need to store indices)
        batch = np.random.choice(len(self.agent.replay_buffer.buffer), size=n_states, replace=False)
        states = [self.agent.replay_buffer.buffer[i][0] for i in batch]  # (state, action, reward, next_state, done)

        confs = []
        greedy_buys = 0
        q_abs = []

        self.agent.q_net.eval()
        with torch.no_grad():
            for s_np in states:
                s = torch.from_numpy(np.asarray(s_np, dtype=np.float32)).unsqueeze(0).to(self.agent.device)
                q = self.agent.q_net(s)[0].detach().cpu().numpy()

                # softmax over actions -> prob
                q2 = q / max(1e-8, temp)
                exps = np.exp(q2 - np.max(q2))
                probs = exps / (np.sum(exps) + 1e-12)

                q_abs.append(float(np.mean(np.abs(q))))

                buy_prob = float(probs[1])  # action 1 = BUY
                confs.append(buy_prob)

                if int(np.argmax(q)) == 1:
                    greedy_buys += 1

        confs = np.asarray(confs, dtype=np.float32)
        return {
            "mean": float(confs.mean()),
            "max": float(confs.max()),
            "min": float(confs.min()),
            "mean_abs_q": float(np.mean(q_abs)),
            "greedy_buy_rate": float(greedy_buys / len(confs)),
        }






