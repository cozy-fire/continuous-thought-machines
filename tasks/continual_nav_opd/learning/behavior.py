"""Interval behavior statistics with complete per-slot episode returns."""
import numpy as np


class IntervalBehavior:
    def __init__(self, slots: int):
        self.episode_returns = np.zeros(slots, dtype=np.float64)
        self.counts = np.zeros(5, dtype=np.int64)
        self.completed = 0
        self.successes = 0
        self.return_sum = 0.

    def add(self, window) -> None:
        self.counts += np.bincount(window.batch.actions.numpy().ravel(), minlength=5)
        rewards = window.rewards.numpy()
        terminated = window.terminated.numpy()
        truncated = window.truncated.numpy()
        for reward, success, timeout in zip(rewards, terminated, truncated):
            self.episode_returns += reward
            done = success | timeout
            # Attribute a FULL episode to the interval containing its terminal step.
            self.completed += int(done.sum())
            self.successes += int((success & done).sum())
            self.return_sum += float(self.episode_returns[done].sum())
            self.episode_returns[done] = 0.

    def flush(self) -> dict:
        total = int(self.counts.sum())
        result = {
            'action_fractions': (self.counts / total).tolist() if total else [0.] * 5,
            # No completed episodes means unknown, not a fabricated zero return/rate.
            'episode_mean_return': self.return_sum / self.completed if self.completed else None,
            'episode_success_rate': self.successes / self.completed if self.completed else None,
        }
        self.counts.fill(0)
        self.completed = self.successes = 0
        self.return_sum = 0.
        # Unfinished episode returns survive log boundaries and rollout windows.
        return result
