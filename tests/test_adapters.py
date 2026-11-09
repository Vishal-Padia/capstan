import json
import os
import sys
import types

import numpy as np

from capstan.adapters import verl_omni, vllm_omni, when_imported


def test_when_imported_patches_on_first_import(tmp_path, monkeypatch):
    (tmp_path / "capstan_fake_module.py").write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    seen = []
    when_imported("capstan_fake_module", lambda m: seen.append(m.value))
    assert seen == []
    import capstan_fake_module  # noqa: F401

    assert seen == [1]
    del sys.modules["capstan_fake_module"]


def test_logprobs_are_written_with_the_step_verl_omni_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTAN_OUT", str(tmp_path))
    compute = verl_omni._corr_metrics(lambda log_prob, rollout_log_prob, timesteps=None: {})
    log = verl_omni._log(lambda self, data, step: None)

    compute(np.array([[0.1]]), np.array([[0.0]]))
    log(None, {"timing_s/gen": 1.5}, 7)

    saved = np.load(tmp_path / f"logprobs_step000007_pid{os.getpid()}.npz")
    assert saved["trainer"].tolist() == [[0.1]] and saved["rollout"].tolist() == [[0.0]]


class _FakeEvent:
    def __init__(self, t):
        self.t = t

    def elapsed_time(self, later):
        return (later.t - self.t) * 1000

    def synchronize(self):
        pass


def test_rollout_split_into_encode_steps_and_decode(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPSTAN_OUT", str(tmp_path))
    times = {"start": 0.0, "diffuse_start": 1.0, "ode": 3.0, "sde": 6.0, "diffuse_end": 6.5, "end": 8.0}
    vllm_omni._record([(label, _FakeEvent(t)) for label, t in times.items()])

    rows = [json.loads(line) for line in (tmp_path / f"phases_{os.getpid()}.jsonl").read_text().splitlines()]
    assert [(r["phase"], r["seconds"]) for r in rows] == [
        ("rollout/encode", 1.0),
        ("rollout/ode_step", 2.0),
        ("rollout/sde_step", 3.0),
        ("rollout/decode", 1.5),
    ]


def test_missing_pipeline_records_nothing_and_says_why(caplog):
    vllm_omni._patch_pipeline(types.SimpleNamespace())
    assert "rollout split not recorded" in caplog.text
