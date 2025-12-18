import multiprocessing as mp
from agents.multi_process.worker import worker_process
import time

class MultiProcessTrainer:
    def __init__(
        self,
        agent,
        env_fn,
        agent_fn,
        n_workers: int = 4,
        steps_per_batch: int = 200,
    ):
        self.agent = agent              # central learner
        self.env_fn = env_fn            # function to create env instances
        self.agent_fn = agent_fn        # function to create worker agents
        self.n_workers = n_workers
        self.steps_per_batch = steps_per_batch
        self.queue = mp.Queue(maxsize=n_workers * 2)
        self.workers: list[mp.Process] = []

    def start_workers(self):
        for i in range(self.n_workers):
            p = mp.Process(
                target=worker_process,
                args=(i, self.env_fn, self.agent_fn, self.queue, self.steps_per_batch),
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

    def train(self, n_batches: int = 2000, updates_per_batch: int = 50):
        """
        n_batches: how many batches we expect from workers
        updates_per_batch: how many DDQN updates we perform per batch
        """
        print(f"[MP Trainer] Starting workers...")
        self.start_workers()

        for batch_idx in range(n_batches):
            batch = self.queue.get()   # wait for worker data

            # Insert transitions into central learner's replay buffer
            for (s, a, r, ns, d) in batch:
                self.agent.push_transition(s, a, r, ns, d)

            # Perform several updates
            for _ in range(updates_per_batch):
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    self.agent.update()

            if (batch_idx + 1) % 10 == 0:
                print(f"[Batch {batch_idx+1}/{n_batches}] Buffer size = {len(self.agent.replay_buffer)}")

        print("[MP Trainer] Training complete.")
