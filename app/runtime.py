"""Atomic JSON output and wall-clock timing; no training hyperparameters."""
import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
import torch


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


class RunTimer:
    """Each invocation gets its own timing entry, including resumed runs.

    Wall-clock durations include data loading. CUDA is synchronized at phase
    boundaries. Total includes initialization, checkpoints and evaluation.
    A hard kill can only leave the last recorded phase, marked running.
    """
    def __init__(self, run, device, enabled=True):
        self.path = run / 'timing.json'
        self.device = device
        self.enabled = enabled
        self.active = False

    def __enter__(self):
        self.started = time.perf_counter()
        self.data = json.loads(self.path.read_text()) if self.enabled and self.path.exists() else {'sessions': []}
        self.session = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), status='running', phases=[])
        self.data['sessions'].append(self.session)
        return self

    def sync(self):
        if self.device == 'cuda' and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    def save(self):
        if self.enabled and self.active and self.path.parent.is_dir():
            self.session['elapsed_seconds'] = time.perf_counter() - self.started
            write_json(self.path, self.data)

    @contextmanager
    def stage(self, name, epoch=None):
        self.sync()
        start = time.perf_counter()
        status = 'failed'
        try:
            yield
            self.sync()
            status = 'complete'
        finally:
            seconds = time.perf_counter() - start
            self.session['phases'].append(dict(stage=name, epoch=epoch, seconds=seconds, status=status))
            self.save()
            print(f'Timing {name}' + (f' epoch {epoch}' if epoch is not None else '') + f': {seconds:.2f} s', flush=True)

    def __exit__(self, exc_type, exc, traceback):
        self.session['status'] = 'complete' if exc_type is None else 'failed'
        self.session['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        self.save()
