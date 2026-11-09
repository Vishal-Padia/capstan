import json

import numpy as np

from capstan import report


def _phase(f, step, phase, seconds, end):
    f.write(json.dumps({"step": step, "phase": phase, "seconds": seconds, "end": end}) + "\n")


def test_profile_and_drift_tables(tmp_path):
    with open(tmp_path / "phases_1.jsonl", "w") as f:
        for step, end in [(1, 100.0), (2, 110.0)]:
            for phase, seconds in [("rollout", 6.0), ("reward", 1.0), ("recompute", 1.0), ("update", 1.0)]:
                _phase(f, step, phase, seconds, end)
            _phase(f, step, "sync", 0.0, end)
            _phase(f, step, "step", 10.0, end)
    with open(tmp_path / "phases_2.jsonl", "w") as f:
        _phase(f, None, "rollout/encode", 1.0, 103.0)
        _phase(f, None, "rollout/ode_step", 4.0, 105.0)
        _phase(f, None, "rollout/sde_step", 1.0, 106.0)
        _phase(f, None, "rollout/decode", 2.0, 95.0)
    np.savez(tmp_path / "logprobs_step000002_pid1.npz", rollout=np.zeros((2, 2)), trainer=np.zeros((2, 2)))
    np.savez(tmp_path / "logprobs_step000002_pid2.npz", rollout=np.zeros((2, 2)), trainer=np.full((2, 2), 1e-3))

    profile = report.profile(tmp_path, skip=1)
    assert "| rollout | 6.00 | 60.0% |" in profile
    assert "| everything else, idle included | 1.00 | 10.0% |" in profile
    assert "| ODE steps | 4.00 | 66.7% | 1 |" in profile
    assert "VAE decode" not in profile

    (tmp_path / "phases_2.jsonl").unlink()
    assert "Not recorded." in report.profile(tmp_path, skip=1)

    drift = report.drift(tmp_path)
    assert "This run failed: 4 of 8 entries over." in drift
    assert "| all | 8 |" in drift
