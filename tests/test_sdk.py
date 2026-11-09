import json
import os

import numpy as np

import capstan


def test_phase_and_logprobs_are_written(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTAN_OUT", str(tmp_path))
    with capstan.phase("rollout", step=1):
        pass
    capstan.record_phase("reward", 2.5, step=1)
    capstan.record_logprobs(rollout=[[0.0, 1.0]], trainer=[[0.5, 1.0]], step=1)

    rows = [json.loads(line) for line in (tmp_path / f"phases_{os.getpid()}.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["rollout", "reward"]
    assert rows[1]["step"] == 1 and rows[1]["seconds"] == 2.5 and rows[1]["end"] > 0

    saved = np.load(tmp_path / f"logprobs_step000001_pid{os.getpid()}.npz")
    assert saved["trainer"].tolist() == [[0.5, 1.0]]
