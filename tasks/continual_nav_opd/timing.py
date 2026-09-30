"""Window-local CUDA stream intervals; synchronous wall timing remains opt-in."""
from contextlib import contextmanager
from contextvars import ContextVar
import time
import torch

_mode = ContextVar('opd_timing_mode', default='events')


@contextmanager
def timing_mode(mode):
    if mode not in ('events', 'synchronized'):
        raise ValueError('timing mode must be events or synchronized')
    token = _mode.set(mode)
    try:
        yield
    finally:
        _mode.reset(token)


def current_timing_mode():
    return _mode.get()


class WindowTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.mode = current_timing_mode()
        self.seconds = {}
        self.events = []

    @contextmanager
    def measure(self, name, *, cpu=False):
        cuda = self.device.type == 'cuda' and not cpu
        if cuda and self.mode == 'events':
            with torch.cuda.device(self.device):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    yield
                finally:
                    end.record()
                    self.events.append((name, start, end))
        else:
            if cuda:
                torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            try:
                yield
            finally:
                if cuda:
                    torch.cuda.synchronize(self.device)
                self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter()-start

    def finish(self):
        # All events use one current stream. One end wait resolves the entire window.
        # Intervals include host gaps between stream work, not just kernel execution.
        if self.events:
            self.events[-1][2].synchronize()
            for name, start, end in self.events:
                self.seconds[name] = self.seconds.get(name, 0.) + start.elapsed_time(end)/1000.
            self.events.clear()
        return dict(self.seconds)
