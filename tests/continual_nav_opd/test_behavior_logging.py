"""Chart whitelist and episode accounting across rollout/log boundaries."""
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
import torch
from tasks.continual_nav_opd.config import load_config
from tasks.continual_nav_opd.learning.behavior import IntervalBehavior
from tasks.continual_nav_opd.wandb_logging import EventLogger, chart_values, TRAIN_METRICS, EVALUATION_METRICS


def window(rewards, terms, truncs, actions):
    return SimpleNamespace(rewards=torch.tensor(rewards), terminated=torch.tensor(terms),
                           truncated=torch.tensor(truncs), batch=SimpleNamespace(actions=torch.tensor(actions)))


class BehaviorLoggingTests(unittest.TestCase):
    def test_completed_training_event_is_printed_and_persisted_without_wandb(self):
        config = load_config('tasks/continual_nav_opd/configs/smoke.yaml')
        event = dict(event='training', stage='pnc/v0/maze_medium/P', phase='P',
                     task='maze_medium', global_env_steps=1600, stage_env_steps=1600,
                     optimizer_updates=4, kl=1.2, total_loss=1.2, agreement=.4,
                     elapsed_seconds=12.)
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()) as stdout:
            logger = EventLogger(root, config, 0)
            logger.emit({'event': 'map_cache'})
            logger.emit(event)
            logger.close()
            self.assertEqual(json.loads(stdout.getvalue()), event)
            from pathlib import Path
            for name in ('events.jsonl', 'metrics.jsonl'):
                rows = [json.loads(line) for line in (Path(root)/name).read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[-1]['kl'], 1.2)

    def test_full_episode_return_survives_flush_and_slots_reset_independently(self):
        stats = IntervalBehavior(2)
        stats.add(window([[1., 2.]], [[False, False]], [[False, True]], [[0, 1]]))
        first = stats.flush()
        self.assertEqual(first['episode_mean_return'], 2.)
        self.assertEqual(first['episode_success_rate'], 0.)
        self.assertEqual(first['action_fractions'], [.5, .5, 0., 0., 0.])
        stats.add(window([[3., 4.], [5., 6.]], [[True, False], [True, False]],
                         [[False, False], [False, True]], [[2, 2], [4, 3]]))
        second = stats.flush()
        self.assertEqual(second['episode_mean_return'], (4.+5.+10.)/3)
        self.assertEqual(second['episode_success_rate'], 2./3)
        self.assertEqual(second['action_fractions'], [0., 0., .5, .25, .25])
        self.assertEqual(stats.episode_returns.tolist(), [0., 0.])

    def test_no_finished_episode_is_missing_not_zero(self):
        stats = IntervalBehavior(1)
        stats.add(window([[2.]], [[False]], [[False]], [[4]]))
        event = {'event':'training', 'phase':'P', **stats.flush()}
        values = chart_values(event)
        self.assertFalse(any(k.endswith('/episode_mean_return') or k.endswith('/episode_success_rate') for k in values))
        stats.add(window([[3.]], [[True]], [[True]], [[0]]))
        final = stats.flush()
        self.assertEqual(final['episode_mean_return'], 5.)
        self.assertEqual(final['episode_success_rate'], 1.)  # One episode, even if both flags are true.

    def test_training_whitelist_and_c_only_ewc(self):
        event = {k: .5 for k in TRAIN_METRICS}
        event.update(event='training', phase='P', ewc=3., seed=0, time_unix=1.,
                     expert_entropy=1., windows=10, reward_sum=10., successes=2,
                     action_counts=[1]*5, action_fractions=[.2]*5)
        values = chart_values(event)
        suffixes = {k.rsplit('/', 1)[-1] for k in values if '/action_fractions/' not in k}
        self.assertEqual(suffixes, TRAIN_METRICS)
        self.assertEqual(sum('/action_fractions/' in k for k in values), 5)
        event['phase'] = 'C'
        self.assertTrue(any(k.endswith('/ewc') for k in chart_values(event)))
        self.assertEqual(chart_values({'event':'memory', 'peak_allocated_bytes':100}), {})

    def test_all_evaluation_metrics_preserved(self):
        for split in ('validation', 'test'):
            event = {k: 1. for k in EVALUATION_METRICS}
            event.update(split=split, action_counts=[1,2,3,4,5], seed=0, kl=2.)
            values = chart_values(event)
            self.assertEqual(len(values), len(EVALUATION_METRICS)+5)
            self.assertFalse(any(k.endswith('/seed') or k.endswith('/kl') for k in values))


if __name__ == '__main__':
    unittest.main()
