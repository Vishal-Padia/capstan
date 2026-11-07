import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

__all__ = ["phase", "record_logprobs", "record_phase"]


def _out() -> Path:
    path = Path(os.environ.get("CAPSTAN_OUT", "capstan_run"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu()
    return np.asarray(x, dtype=np.float64)


def record_phase(name, seconds, step=None):
    with open(_out() / "phases.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "phase": name, "seconds": float(seconds)}) + "\n")


# Wall-clock time: the block must end on a sync point (e.g. results copied to CPU), or GPU work leaks into the next one.
@contextmanager
def phase(name, step=None):
    start = time.perf_counter()
    try:
        yield
    finally:
        record_phase(name, time.perf_counter() - start, step)


def record_logprobs(rollout, trainer, step=None, timesteps=None):
    arrays = {"rollout": _numpy(rollout), "trainer": _numpy(trainer)}
    if timesteps is not None:
        arrays["timesteps"] = _numpy(timesteps)
    if step is not None:
        arrays["step"] = np.asarray(step)
    out = _out()
    index = len(list(out.glob("logprobs_*.npz")))
    np.savez(out / f"logprobs_{index:05d}.npz", **arrays)
