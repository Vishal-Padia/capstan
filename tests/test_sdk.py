import json

import numpy as np

import capstan


def test_phase_and_logprobs_are_written(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTAN_OUT", str(tmp_path))
    with capstan.phase("rollout", step=1):
        pass
    capstan.record_phase("reward", 2.5, step=1)
    capstan.record_logprobs(rollout=[[0.0, 1.0]], trainer=[[0.5, 1.0]], step=1)
    capstan.record_logprobs(rollout=[[0.0]], trainer=[[0.0]])

    rows = [json.loads(line) for line in (tmp_path / "phases.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["rollout", "reward"]
    assert rows[1] == {"step": 1, "phase": "reward", "seconds": 2.5}

    first = np.load(tmp_path / "logprobs_00000.npz")
    assert first["trainer"].tolist() == [[0.5, 1.0]] and int(first["step"]) == 1
    assert (tmp_path / "logprobs_00001.npz").exists()
